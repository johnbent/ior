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
    IOD_GetFileSize
};

typedef struct iod_parameters_s {
	int checksum;
} iod_parameters_t;

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
} iod_state_t;

static iod_state_t *istate = NULL; 

/***************************** F U N C T I O N S ******************************/

#define IDEBUG(rank, format, ...)                     \
do {                                    \
    int _rank = (rank);                         \
                                    \
    if (_rank == 0) {                          \
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

static void IOD_Barrier(int rank, MPI_Comm mcom) {
    IDEBUG(rank, "MPI_Barrier");
    MPI_Barrier(mcom);
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
open_rd(iod_state_t *s, char *filename, MPI_Comm mcom) {
	iod_ret_t ret = 0;

	/* start the read trans */
	if (s->myrank == 0) {
		ret=iod_trans_start(s->coh, &(s->tid),NULL,0,IOD_TRANS_R,NULL);
		IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);
	}
	IOD_Barrier(s->myrank, mcom);

	ret = iod_obj_open_read(s->coh, s->oid, s->tid, NULL, &(s->oh), NULL);
	IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);
	return ret;
}

int
open_wr(iod_state_t *I, char *filename, MPI_Comm mcom) {
	iod_parameters_t *P = &(I->params);
	int ret = 0;

	/* start the write trans */
	I->tid++;
	if (I->myrank == 0) {
		ret = iod_trans_start(I->coh, &(I->tid), NULL, 0, IOD_TRANS_W, NULL);
		IOD_RETURN_ON_ERROR("iod_trans_start", ret);
		IDEBUG(I->myrank, "iod_trans_start %d : success", I->tid );
	}
	MPI_Barrier(mcom);

	/* create the obj */
	/* only rank 0 does create then a barrier */
	/* alternatively everyone does create, most fail with EEXIST, no barrier */
	I->otype = IOD_OBJ_BLOB;
	I->oid = 0;
	IOD_OBJID_SETOWNER_APP(I->oid)
	IOD_OBJID_SETTYPE(I->oid, IOD_OBJ_BLOB);
	if( !I->myrank ) {
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
	MPI_Barrier(mcom);

	/* now open the obj */
	ret = iod_obj_open_write(I->coh, I->oid, I->tid, NULL, &(I->oh), NULL);
	IOD_RETURN_ON_ERROR("iod_obj_open_write", ret);
	return ret;	
}

static iod_state_t * Init(IOR_param_t *param)
{
    int rc;
    iod_state_t *istate;
    istate = malloc(sizeof(iod_state_t));
    DCHECK(istate==NULL?-1:0,"malloc failed");

    istate->mcom = param->testComm;
    MPI_Comm_rank(istate->mcom, &(istate->myrank));
    MPI_Comm_size(istate->mcom, &(istate->nranks));
    IDEBUG(rank,"About to init");
    rc = iod_initialize(istate->mcom, NULL, istate->nranks, istate->nranks);
    IDEBUG(rank,"Done with init");
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

    /*
    con_open_hint = (iod_hint_list_t *)malloc(sizeof(iod_hint_list_t) + sizeof(iod_hint_t));
    con_open_hint->num_hint = 1;
    con_open_hint->hint[0].key = "iod_hint_co_scratch_cksum";
    */

    /* open and create the container here */
    IDEBUG(istate->myrank, "About to open container %s with %d ranks", 
                            testFileName, istate->nranks );
    rc = iod_container_open(testFileName, con_open_hint, 
        IOD_CONT_CREATE|IOD_CONT_RW, &(istate->coh), NULL);
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


/*
 * Open a file through the IOD interface.
 */
static void *IOD_Open(char *testFileName, IOR_param_t * param)
{
    int rc;
    int rank;
    MPI_Comm_rank(param->testComm,&rank);

    if (!istate) {
        istate = Init(param);
    };

    rc = ContainerOpen(testFileName, param, istate);
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);

    if (param->open == WRITE) {
        rc = SkipTidZero(istate,testFileName);
        DCHECK(rc, "%s:%d", __FILE__, __LINE__);
        IOD_Barrier(istate->myrank,istate->mcom);

        rc = open_wr(istate, testFileName, istate->mcom);
        DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    } else {
        assert(param->open == READ);
        rc = open_rd(istate, testFileName, istate->mcom);
    }
    IOD_Barrier(istate->myrank, istate->mcom);
}

/*
 * Write or read access to file using the IOD interface.
 */
static IOR_offset_t IOD_Xfer(int access, void *file, IOR_size_t * buffer,
                   IOR_offset_t length, IOR_param_t * param)
{
    DCHECK(-ENOSYS,"%s:%d", __FILE__, __LINE__);
    int xferRetries = 0;
    long long remaining = (long long)length;
    char *ptr = (char *)buffer;
    long long rc;
    int fd;

    fd = *(int *)file;

    /* seek to offset */
    if (lseek64(fd, param->offset, SEEK_SET) == -1)
        ERR("lseek64() failed");

    while (remaining > 0) {
        /* write/read file */
        if (access == WRITE) {  /* WRITE */
            if (verbose >= VERBOSE_4) {
                fprintf(stdout,
                    "task %d writing to offset %lld\n",
                    rank,
                    param->offset + length - remaining);
            }
            rc = write(fd, ptr, remaining);
            if (rc == -1)
                ERR("write() failed");
            if (param->fsyncPerWrite == TRUE)
                IOD_Fsync(&fd, param);
        } else {    /* READ or CHECK */
            if (verbose >= VERBOSE_4) {
                fprintf(stdout,
                    "task %d reading from offset %lld\n",
                    rank,
                    param->offset + length - remaining);
            }
            rc = read(fd, ptr, remaining);
            if (rc == 0)
                ERR("read() returned EOF prematurely");
            if (rc == -1)
                ERR("read() failed");
        }
        if (rc < remaining) {
            fprintf(stdout,
                "WARNING: Task %d, partial %s, %lld of %lld bytes at offset %lld\n",
                rank,
                access == WRITE ? "write()" : "read()",
                rc, remaining,
                param->offset + length - remaining);
            if (param->singleXferAttempt == TRUE)
                MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, -1),
                      "barrier error");
            if (xferRetries > MAX_RETRY)
                ERR("too many retries -- aborting");
        }
        assert(rc >= 0);
        assert(rc <= remaining);
        remaining -= rc;
        ptr += rc;
        xferRetries++;
    }
    return (length);
}

/*
 * Perform fsync().
 */
static void IOD_Fsync(void *fd, IOR_param_t * param)
{
    if (fsync(*(int *)fd) != 0)
        EWARN("fsync() failed");
}

/*
 * Close a file through the IOD interface.
 */
static void IOD_Close(void *fd, IOR_param_t * param)
{
    if (close(*(int *)fd) != 0)
        ERR("close() failed");
    free(fd);
}

/*
 * Delete a file through the IOD interface.
 */
static void IOD_Delete(char *testFileName, IOR_param_t * param)
{
    char errmsg[256];
    sprintf(errmsg, "[RANK %03d]: unlink() of file \"%s\" failed\n",
        rank, testFileName);
    if (unlink(testFileName) != 0)
        EWARN(errmsg);
}

/*
 * Determine api version.
 */
static void IOD_SetVersion(IOR_param_t * test)
{
    strcpy(test->apiVersion, test->api);
}

/*
 * Use IOD stat() to return aggregate file size.
 */
static IOR_offset_t IOD_GetFileSize(IOR_param_t * test, MPI_Comm testComm,
                      char *testFileName)
{
    struct stat stat_buf;
    IOR_offset_t aggFileSizeFromStat, tmpMin, tmpMax, tmpSum;

    if (stat(testFileName, &stat_buf) != 0) {
        ERR("stat() failed");
    }
    aggFileSizeFromStat = stat_buf.st_size;

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
