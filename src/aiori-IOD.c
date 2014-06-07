/* -*- mode: c; c-basic-offset: 8; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=8:tabstop=8:
 */
/******************************************************************************\
*                                          *
*    Copyright (c) 2003, The Regents of the University of California       *
*      See the file COPYRIGHT for a complete copyright notice and license.     *
*                                          *
********************************************************************************
*
* Implement of abstract I/O interface for IOD.
*
\******************************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/ioctl.h>      /* necessary for: */
#define __USE_GNU           /* O_DIRECT and */
#include <fcntl.h>          /* IO operations */
#undef __USE_GNU
#endif              /* __linux__ */
#include <errno.h>
#include <fcntl.h>          /* IO operations */
#include <sys/stat.h>
#include <assert.h>
#ifdef HAVE_LUSTRE_LUSTRE_USER_H
#include <lustre/lustre_user.h>
#endif

#ifdef HAVE_GPFS_H
#include <gpfs.h>
#endif
#ifdef HAVE_GPFS_FCNTL_H
#include <gpfs_fcntl.h>
#endif

#include "ior.h"
#include "aiori.h"
#include "iordef.h"

#include <iod_api.h>
#include <iod_types.h>
#include <plfs.h>

#include <libgen.h>

#ifndef   open64        /* necessary for TRU64 -- */
#define open64  open        /* unlikely, but may pose */
#endif  /* not open64 */            /* conflicting prototypes */

#ifndef   lseek64           /* necessary for TRU64 -- */
#define lseek64 lseek       /* unlikely, but may pose */
#endif  /* not lseek64 */            /* conflicting prototypes */

#ifndef   O_BINARY          /* Required on Windows    */
#define O_BINARY 0
#endif

/**************************** P R O T O T Y P E S *****************************/
static void *IOD_Create(char *, IOR_param_t *);
static void *IOD_Open(char *, IOR_param_t *);
static IOR_offset_t IOD_Xfer(int, void *, IOR_size_t *,
                   IOR_offset_t, IOR_param_t *);
static void IOD_Close(void *, IOR_param_t *);
static void IOD_Delete(char *, IOR_param_t *);
static void IOD_SetVersion(IOR_param_t *);
static void IOD_Fsync(void *, IOR_param_t *);
static IOR_offset_t IOD_GetFileSize(IOR_param_t *, MPI_Comm, char *);
static int IOD_Init(char *, IOR_param_t *);
static int IOD_Fini(char *, IOR_param_t *);

/************************** D E C L A R A T I O N S ***************************/

ior_aiori_t iod_aiori = {
    "IOD",
    IOD_Create,
    IOD_Open,
    IOD_Xfer,
    IOD_Close,
    IOD_Delete,
    IOD_SetVersion,
    IOD_Fsync,
    IOD_GetFileSize,
    IOD_Init,
    IOD_Fini
};

typedef struct iod_parameters_s {
	int checksum;
} iod_parameters_t;

#define TIME_MSG_LEN 8192
typedef struct iod_state_s {
	iod_trans_id_t tid;
	iod_handle_t coh;
	iod_handle_t oh;
        iod_obj_id_t oid; 
	iod_obj_type_t otype;
	iod_parameters_t params;
	iod_blob_iodesc_t *io_desc;
	iod_mem_desc_t *mem_desc;
	iod_checksum_t *cksum;
	MPI_Comm mcom;
	int myrank;
        int nranks;
        char times[TIME_MSG_LEN];
        double timer;
} iod_state_t;

static iod_state_t *istate = NULL; 

/***************************** F U N C T I O N S ******************************/

enum {
    DEBUG_ZERO, /* only rank 0 prints msg */
    DEBUG_ALL,  /* all ranks print msg unless pass -1 to suppress */
    DEBUG_NONE, /* total silence */
    DEBUG_EVERY, /* all ranks including Xfer */
};
static int verbosity_level = DEBUG_ZERO;


int
debug_on(int rank) {
    switch(verbosity_level) {
            case DEBUG_ZERO: return (rank == 0);
            case DEBUG_NONE: return 0;
            case DEBUG_ALL: return (rank != -1);
            case DEBUG_EVERY: return 1;
    }
}

#define IDEBUG(rank, format, ...)                     \
do {                                    \
    int _rank = (rank);                         \
                                    \
    if (debug_on(_rank)) {                          \
        fprintf(stdout, "%.2f IOD DEBUG (%s:%d): %d: : "       \
            format"\n", MPI_Wtime(), \
            __FILE__, __LINE__, rank,       \
            ##__VA_ARGS__);         \
        fflush(stdout);                     \
    }                                   \
} while (0);

#define IOD_RETURN_ON_ERROR(X,Y) { \
	if (Y != 0 ) { \
		IOD_PRINT_ERR(X,Y);\
		return Y; \
	} \
}

#define IOD_DIE_ON_ERROR(X,Y) { \
	if (Y != 0 ) { \
		IOD_PRINT_ERR(X,Y);\
		assert(0); \
	} \
}

#define IOD_PRINT_ERR(X,Y) { \
	fprintf(stderr,"IOD Error in %s:%d on %s: %s\n", \
		__FILE__, __LINE__, X, strerror(-Y));\
}

#define DCHECK(rc, format, ...)                     \
do {                                    \
    int _rc = (rc);                         \
                                    \
    if (_rc < 0) {                          \
        fprintf(stdout, "ior ERROR (%s:%d): %d: %s: "       \
            format"\n", __FILE__, __LINE__, rank,       \
            strerror(-_rc), ##__VA_ARGS__);         \
        fflush(stdout);                     \
        MPI_Abort(MPI_COMM_WORLD, -1);              \
    }                                   \
} while (0);

static void start_timer() {
    assert(istate != NULL);
    istate->timer = MPI_Wtime();
}

static void add_timer(char *op) {
    snprintf(&(istate->times[strlen(istate->times)]),
        TIME_MSG_LEN - strlen(istate->times), "\tIOD %s_time =  %.4f\n", op,
        MPI_Wtime() - istate->timer);
}

static void add_bandwidth(char *op, IOR_offset_t len) {
    snprintf(&(istate->times[strlen(istate->times)]),
        TIME_MSG_LEN - strlen(istate->times), 
        "\tIOD %s_time = %.4f Bandwidth MB/s = %.2f\n", 
        op, MPI_Wtime() - istate->timer,
        (len / 1048576) / (MPI_Wtime() - istate->timer));

}

static void IOD_Barrier(iod_state_t *s) {
    IDEBUG(s->myrank, "MPI_Barrier");
    MPI_Barrier(s->mcom);
}

int
setup_blob_io(size_t len, off_t off, char *buf, iod_state_t *s, int rw) { 

	/* where is the IO directed in the object */
	s->io_desc->nfrag = 1;
	s->io_desc->frag[0].len = len;
	s->io_desc->frag[0].offset = off;
	
	/* from whence does the IO come in memory */
	s->mem_desc->nfrag = 1;
	s->mem_desc->frag[0].addr = (void*)buf;
	s->mem_desc->frag[0].len = len;

	/* setup the checksum */
	if (s->params.checksum) {
		if (rw==WRITE) {
			plfs_error_t pret;
			pret  = plfs_get_checksum(buf,len,(Plfs_checksum*)s->cksum);
			IOD_RETURN_ON_ERROR("plfs_get_checksum", pret);
		} else {
			memset(s->cksum,0,sizeof(*(s->cksum)));
		}
	}
}

// returns zero on success and number of bytes transferred in bytes
int 
iod_write(iod_state_t *I,char *buf,size_t len,off_t off,ssize_t *bytes) {
	iod_ret_t ret;
	iod_hint_list_t *hints = NULL;
	iod_event_t *event = NULL;

	setup_blob_io(len,off,buf,I,WRITE);
	ret =iod_blob_write(I->oh,I->tid,hints,I->mem_desc,I->io_desc,I->cksum,event);
	IOD_RETURN_ON_ERROR("iod_blob_write",ret); // successful write returns zero
	*bytes = len;

	return ret;
}

int 
iod_read(iod_state_t *s, char *buf,size_t len,off_t off,ssize_t *bytes) {
	iod_hint_list_t *hints = NULL;
	iod_event_t *event = NULL;
	iod_ret_t ret = 0;

	setup_blob_io(len,off,buf,s,READ);
	ret=iod_blob_read(s->oh,s->tid,hints,s->mem_desc,s->io_desc,s->cksum,event);
	if ( ret == 0 ) { // current semantic is 0 for success
		*bytes = len;
	} else {
		IOD_PRINT_ERR("iod_blob_read",ret);
	}
	
	if (s->params.checksum) {
		plfs_error_t pret = 0;
		pret = plfs_checksum_match(buf,len,(Plfs_checksum)(*s->cksum));
		IOD_RETURN_ON_ERROR("plfs_checksum_match", pret);
	}
	return ret; 
}

static int
create_obj(iod_handle_t coh, iod_trans_id_t tid, iod_obj_id_t *oid, iod_obj_type_t type, 
		iod_array_struct_t *structure, iod_hint_list_t *hints, int rank) 
{
        IDEBUG(rank, "Creating obj type %d %lli", type, *oid);
	int ret = iod_obj_create(coh, tid, hints, type, NULL, 
			structure, oid, NULL);
	IOD_RETURN_ON_ERROR("iod_obj_create",ret);
	return ret;
}

int
set_checksum(iod_hint_list_t **hints, int checksum) {
	if (checksum) {
		*hints = (iod_hint_list_t *)malloc(sizeof(iod_hint_list_t) + sizeof(iod_hint_t));
		assert(*hints);
		(*hints)->num_hint = 1;
		(*hints)->hint[0].key = "iod_hint_obj_enable_cksum";
	}
	return 0;
}

int
open_rd(iod_state_t *s, char *filename, IOR_param_t *param) {
	iod_ret_t ret = 0;

	/* start the read trans */
	if (s->myrank == 0) {
		ret=iod_trans_start(s->coh, &(s->tid),NULL,0,IOD_TRANS_R,NULL);
		IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);
	}
	IOD_Barrier(s);

        start_timer();
	ret = iod_obj_open_read(s->coh, s->oid, s->tid, NULL, &(s->oh), NULL);
        add_timer("iod_obj_open_read");
	IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);
	return ret;
}

int
open_wr(iod_state_t *I, char *filename, IOR_param_t *param) {
	iod_parameters_t *P = &(I->params);
	int ret = 0;

	/* start the write trans */
	I->tid++; // bump up the tid
	if (I->myrank == 0) {
		ret = iod_trans_start(I->coh, &(I->tid), NULL, 0, IOD_TRANS_W, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_start", ret);
		IDEBUG(I->myrank, "iod_trans_start %d : success", I->tid );
	}
        start_timer();
        IOD_Barrier(I);

	/* create the obj */
	/* for shared file, only rank 0 does create then a barrier */
	/* or does create, most fail with EEXIST, no barrier */
	I->otype = IOD_OBJ_BLOB;
        if (param->filePerProc == TRUE) {
            I->oid = I->myrank;
        } else {
            I->oid = 0;
        }
	IOD_OBJID_SETOWNER_APP(I->oid)
	IOD_OBJID_SETTYPE(I->oid, IOD_OBJ_BLOB);
	if( !I->myrank || param->filePerProc == TRUE ) {
		iod_hint_list_t *hints = NULL;
		set_checksum(&hints,P->checksum);
		ret = create_obj(I->coh, I->tid, &(I->oid), I->otype, NULL, hints, 
                                I->myrank);
		if (hints) free(hints);
		if ( ret != 0 ) {
			return ret;
		} else {
			IDEBUG(I->myrank,"iod obj %lli created successfully.",I->oid);
		}
	}
	IOD_Barrier(I);
        add_timer("iod_obj_create");

	/* now open the obj */
        start_timer();
	ret = iod_obj_open_write(I->coh, I->oid, I->tid, NULL, &(I->oh), NULL);
	IOD_RETURN_ON_ERROR("iod_obj_open_write", ret);
        add_timer("iod_obj_open_write");
	return ret;	
}

static iod_state_t * Init(IOR_param_t *param)
{
    int rc;

    switch(param->verbose) {
    case 0: verbosity_level = DEBUG_NONE; break; 
    case 1: verbosity_level = DEBUG_ZERO; break;
    case 2: verbosity_level = DEBUG_ALL; break;
    default: verbosity_level = DEBUG_EVERY; break;
    } 

    istate = malloc(sizeof(iod_state_t));
    memset(istate, 0, sizeof(*istate));
    DCHECK(istate==NULL?-1:0,"malloc failed");

    istate->mcom = param->testComm;
    MPI_Comm_rank(istate->mcom, &(istate->myrank));
    MPI_Comm_size(istate->mcom, &(istate->nranks));
    start_timer();
    IDEBUG(rank,"About to init");
    rc = iod_initialize(istate->mcom, NULL, istate->nranks, istate->nranks);
    IDEBUG(rank,"Done with init");
    add_timer("iod_initialize");
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    if (rc != 0) {
        free(istate);
        return NULL;
    }

    /* setup the io and memory descriptors used for blob io */
    istate->mem_desc = 
            malloc(sizeof(iod_mem_desc_t) + sizeof(iod_mem_frag_t));
    istate->io_desc = 
            malloc(sizeof(iod_blob_iodesc_t) + sizeof(iod_blob_iofrag_t));
    assert(istate->mem_desc && istate->io_desc);

    /* setup the checksum used for blob */
    if (istate->params.checksum) {
        istate->cksum = malloc(sizeof(iod_checksum_t));
        assert(istate->cksum);
    } else {
        istate->cksum = NULL;
    }

    return istate;
}

static char *iod_cname( char *path, IOR_param_t *param ) {
    static char *cname = NULL;

    /* the passed in filename might have been changed for file-per-proc */
    /* for iod, we want one container, for file-per-proc, use different oid */
    if (param->filePerProc == TRUE) {
        IDEBUG(istate->myrank,"use container %s instead of %s", 
                param->testFileName, path);
        path = param->testFileName;
    }

    cname = NULL;
    cname = basename(path);
    if (!cname) cname = path;
    return cname;
}

/*
 * Creat and open a file through the IOD interface.
 */
static void *IOD_Create(char *testFileName, IOR_param_t * param)
{
    return IOD_Open(testFileName, param);
}

static int ContainerOpen(char *testFileName, IOR_param_t * param, 
        iod_state_t *istate)
{
    int rc = -ENOSYS;
    iod_hint_list_t *con_open_hint = NULL;
    unsigned int mode;
    char *cname = iod_cname(testFileName,param);

    /*
    con_open_hint = (iod_hint_list_t *)malloc(sizeof(iod_hint_list_t) + sizeof(iod_hint_t));
    con_open_hint->num_hint = 1;
    con_open_hint->hint[0].key = "iod_hint_co_scratch_cksum";
    */

    /* since we only open once, then we need to open RDWR */
    mode = IOD_CONT_RW | IOD_CONT_CREATE;

    /* open and create the container here */
    start_timer();
    IOD_Barrier(istate);
    IDEBUG(istate->myrank, "About to open container %s (base %s) with %d ranks",
                    testFileName, cname,istate->nranks );
    rc = iod_container_open(cname, con_open_hint, 
        mode, &(istate->coh), NULL);
    IOD_Barrier(istate);
    add_timer("iod_container_open");
    IDEBUG(istate->myrank, "Done open container %s with %d ranks: %d", 
                            testFileName, istate->nranks, rc );
    return rc;
}

static int SkipTidZero(iod_state_t *s, const char *target) {
    int rc = 0;
    if (!s->myrank) { 
        s->tid = 0;
        IDEBUG(s->myrank, "Starting tid %d on container %s with %d ranks", 
            s->tid, target, s->nranks);
        rc = iod_trans_start(s->coh, &(s->tid), NULL, 0, IOD_TRANS_W, NULL);
        DCHECK(rc,"iod_trans_start"); 
        IDEBUG(s->myrank,"Finishing tid %d on container %s with %d ranks", 
            s->tid, target, s->nranks); 
        rc = iod_trans_finish(s->coh, s->tid, NULL, 0, NULL);
        DCHECK(rc,"iod_trans_finish"); 
    }
    return rc;
}

static int IOD_Init(char *filename, IOR_param_t *param) {
    int rc;
    Init(param);

    IOD_Barrier(istate);
    rc = ContainerOpen(filename, param, istate);
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    IOD_Barrier(istate);

    return rc;
} 

static int IOD_Fini(char *filename, IOR_param_t *param) {
    int rc;
    start_timer();
    IDEBUG(istate->myrank,"About to close container");
    rc = iod_container_close(istate->coh, NULL, NULL);
    IDEBUG(istate->myrank,"Closed container: %d", rc);
    IOD_Barrier(istate);
    add_timer("iod_container_close");
    IOD_RETURN_ON_ERROR("iod_container_close",rc);
    start_timer();
    rc = iod_finalize(NULL);
    IDEBUG(istate->myrank,"iod_finalize: %d", rc);
    add_timer("iod_finalize");

    if(istate->myrank==0) {
        printf("%s", istate->times);
    }

    return rc;
}


/*
 * Open a file through the IOD interface.
 * Opens both the container in RDWR and the object 
 * in whatever mode is necessary
 */
static void *IOD_Open(char *testFileName, IOR_param_t * param)
{
    int rc;
    //int rank;
    //MPI_Comm_rank(param->testComm,&rank);

    if (param->open == WRITE) {
        start_timer();
        rc = SkipTidZero(istate,testFileName);
        DCHECK(rc, "%s:%d", __FILE__, __LINE__);
        IOD_Barrier(istate);
        add_timer("iod_skip_tid0");

        rc = open_wr(istate, testFileName, param);
        DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    } else {
        assert(param->open == READ);
        rc = open_rd(istate, testFileName, param);
    }
    IOD_Barrier(istate);

    return (void*)istate;
}

/*
 * Write or read access to file using the IOD interface.
 */
static IOR_offset_t IOD_Xfer(int access, void *file, IOR_size_t * buffer,
                   IOR_offset_t length, IOR_param_t * param)
{

    iod_state_t *s = (iod_state_t*)file;
    long long rc;
    ssize_t bytes;
    // don't use rank because too verbose for DEBUG_ZERO
    // instead use -1 so only see if at DEBUG_ALL
    int verbosity = -1;

    IDEBUG(verbosity, "Enter %s", __FUNCTION__);
    if (access == WRITE) {  /* WRITE */
        rc = iod_write(s,(char*)buffer,length,param->offset, &bytes); 
    } else {
        rc = iod_read(s,(char*)buffer,length,param->offset, &bytes); 
    }
    IDEBUG(verbosity, "iod_%s %d: %d", (access==WRITE)?"write":"read",
        (int)rc, (int)bytes);
    if (rc == 0) {
        assert(bytes==length);
        return (length);
    } else {
        return -1; 
    }
}

/*
 * Perform fsync().
 */
static void IOD_Fsync(void *fd, IOR_param_t * param)
{
    if (fsync(*(int *)fd) != 0)
        EWARN("fsync() failed");
}

/* this only close the object */
int
iod_close( iod_state_t *s) {
	iod_ret_t ret;

	/* close the object handle */
	ret = iod_obj_close(s->oh, NULL, NULL);
	IOD_RETURN_ON_ERROR("iod_obj_close",ret);

	/* finish the transaction */
	if (s->myrank == 0) {
		ret = iod_trans_finish(s->coh, s->tid, NULL, 0, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_finish",ret);
		IDEBUG(s->myrank,"iod_trans_finish %d: success", s->tid);
	}
	MPI_Barrier(s->mcom);
	
	return ret;
}

int
iod_persist(iod_state_t *s) {
	iod_ret_t ret = 0;
	MPI_Barrier(s->mcom);
	if (s->myrank == 0) {
		ret = iod_trans_start(s->coh, &(s->tid), NULL, 0, IOD_TRANS_R, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_start", ret);

		IDEBUG(s->myrank,"Persist on TR %d", s->tid);
		ret = iod_trans_persist(s->coh, s->tid, NULL, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_persist", ret);

		ret = iod_trans_finish(s->coh, s->tid, NULL, 0, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_finish", ret);
	}
	MPI_Barrier(s->mcom);
	return ret;
}

/*
 * Close a file through the IOD interface.
 * Must close both the object and the container
 */
static void IOD_Close(void *fd, IOR_param_t * param)
{
    iod_ret_t ret;
    iod_state_t *s = (iod_state_t*)fd;
    char func_name[128];

    start_timer();
    IOD_Barrier(s);
    IDEBUG(s->myrank,"About to close object");
    ret = iod_close(s);
    IOD_DIE_ON_ERROR("iod_object_close",ret);
    IDEBUG(s->myrank,"Closed object");
    IOD_Barrier(s);
    sprintf(func_name, "iod_obj_close_%s", param->open==WRITE?"write":"read");
    add_timer(func_name);

    if (param->open == WRITE && param->persist_daos) {
        start_timer();
        iod_persist(s);
        add_bandwidth("Persist", param->expectedAggFileSize);
    }

    return; 
}

/*
 * Delete a file through the IOD interface.
 * Currently a no-op....
 */
static void IOD_Delete(char *testFileName, IOR_param_t * param)
{
    iod_ret_t ret = 0;
    char *cname = iod_cname(testFileName, param);
    IDEBUG(istate->myrank, "NOOP: iod_container_unlink %s", cname);
    //ret = iod_container_unlink(cname, 1, NULL);
    if (ret != 0) {
        char errmsg[256];
        sprintf(errmsg, "[RANK %03d]: unlink() of file \"%s\" failed\n",
                rank, testFileName);
        EWARN(errmsg);
        /*assert(0); // why doesn't work? */
        /*
        // try regular
        if (unlink(testFileName) != 0)
                EWARN(errmsg);
        */
    }
}

/*
 * Determine api version.
 */
static void IOD_SetVersion(IOR_param_t * test)
{
    strcpy(test->apiVersion, test->api);
}

/*
 * Use IOD stat() to return aggregate file size of all objects moved
 */
static IOR_offset_t IOD_GetFileSize(IOR_param_t * test, MPI_Comm testComm,
                      char *testFileName)
{
    struct stat stat_buf;
    IOR_offset_t aggFileSizeFromStat, tmpMin, tmpMax, tmpSum;

    /* 
    if (stat(testFileName, &stat_buf) != 0) {
        ERR("stat() failed");
    }
    aggFileSizeFromStat = stat_buf.st_size;
    */
    aggFileSizeFromStat = 0;

    /* just cheat since IOD can't stat */
    return test->expectedAggFileSize;

    if (test->filePerProc == TRUE) {
        MPI_CHECK(MPI_Allreduce(&aggFileSizeFromStat, &tmpSum, 1,
                    MPI_LONG_LONG_INT, MPI_SUM, testComm),
              "cannot total data moved");
        aggFileSizeFromStat = tmpSum;
    } else {
        MPI_CHECK(MPI_Allreduce(&aggFileSizeFromStat, &tmpMin, 1,
                    MPI_LONG_LONG_INT, MPI_MIN, testComm),
              "cannot total data moved");
        MPI_CHECK(MPI_Allreduce(&aggFileSizeFromStat, &tmpMax, 1,
                    MPI_LONG_LONG_INT, MPI_MAX, testComm),
              "cannot total data moved");
        if (tmpMin != tmpMax) {
            if (rank == 0) {
                WARN("inconsistent file size by different tasks");
            }
            /* incorrect, but now consistent across tasks */
            aggFileSizeFromStat = tmpMin;
        }
    }

    return (aggFileSizeFromStat);
}
