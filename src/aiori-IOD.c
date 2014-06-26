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
#include "utilities.h"

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

typedef struct iod_state_s {
	iod_trans_id_t tid;
	iod_trans_id_t tag; // if we fetch
	iod_handle_t coh;
	iod_handle_t oh;
        iod_obj_id_t oid; 
	iod_obj_type_t otype;
	iod_blob_iodesc_t *io_desc;
        iod_kv_t kv;
        iod_array_struct_t *array;
        iod_hyperslab_t *slab;
	iod_mem_desc_t *mem_desc;
	iod_checksum_t *cksum;
        int checksum; // are we doing checksums?
	MPI_Comm mcom;
	int myrank;
        int nranks;
        int ssf; 
} iod_state_t;

static iod_state_t *istate = NULL; 

#define WRITE_MODE 1
#define READ_MODE 0

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

static void IOD_Barrier(iod_state_t *s) {
    IDEBUG(s->myrank, "MPI_Barrier");
    MPI_Barrier(s->mcom);
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
    StartTimer();
    IOD_Barrier(istate);
    IDEBUG(istate->myrank, "About to open container %s (base %s) with %d ranks",
                    testFileName, cname,istate->nranks );
    rc = iod_container_open(cname, con_open_hint, 
        mode, &(istate->coh), NULL);
    IOD_Barrier(istate);
    AddTimer("iod_container_open");
    IDEBUG(istate->myrank, "Done open container %s with %d ranks: %d", 
                            testFileName, istate->nranks, rc );
    return rc;
}

void
setup_mem_desc(iod_state_t *s, char *buf, off_t len) {
    /* from whence does the IO come in memory */
    s->mem_desc->nfrag = 1;
    s->mem_desc->frag[0].addr = (void*)buf;
    s->mem_desc->frag[0].len = len;
}

void
setup_blob_io(size_t len, off_t off, char *buf, iod_state_t *s) { 
	/* where is the IO directed in the object */
	s->io_desc->nfrag = 1;
	s->io_desc->frag[0].len = len;
	s->io_desc->frag[0].offset = off;
        
        setup_mem_desc(s, buf, len);
}

void
setup_kv_io(size_t len, off_t *off, char *buf, iod_state_t *s) {
    s->kv.key = (void *)off;
    s->kv.key_len = (iod_size_t)sizeof(*off);
    s->kv.value = (void*)buf;
    s->kv.value_len = (iod_size_t)len;
}

int
setup_array(iod_state_t *I, size_t io_size, size_t last_cell) {
    IDEBUG(I->myrank, "Setting up array");
    int num_dims = 1;   /* 1D for now */
    /* make the array structure */
    I->array = (iod_array_struct_t *)malloc(sizeof(iod_array_struct_t));
    assert(I->array);
    I->array->cell_size = io_size;
    I->array->num_dims = num_dims; 
    I->array->firstdim_max    = IOD_DIMLEN_UNLIMITED;
    I->array->chunk_dims = NULL;
    I->array->current_dims = (iod_size_t*)malloc(sizeof(iod_size_t) * num_dims);
    assert(I->array->current_dims);
    
    I->array->current_dims[0] = last_cell;
    IDEBUG(I->myrank, "Current dim is %ld", I->array->current_dims[0]);

    /* make the hyperslab structure */
    I->slab = (iod_hyperslab_t*)malloc(sizeof(iod_hyperslab_t));
    I->slab->start  = (iod_size_t*)malloc(sizeof(iod_size_t)*num_dims);
    I->slab->count  = (iod_size_t*)malloc(sizeof(iod_size_t)*num_dims);
    I->slab->stride = (iod_size_t*)malloc(sizeof(iod_size_t)*num_dims);
    I->slab->block  = (iod_size_t*)malloc(sizeof(iod_size_t)*num_dims);
}

int
teardown_array(iod_state_t *I) {
    if (I->array) {
        free(I->array->current_dims);
        free(I->array);
    }
    if (I->slab) {
        free(I->slab->start);
        free(I->slab->count);
        free(I->slab->stride);
        free(I->slab->block);
        free(I->slab);
    }
}

void
setup_array_io(size_t len, off_t off, char *buf, iod_state_t *s,size_t cell_sz) { 
    size_t which_cell = off / cell_sz;
    size_t num_cells = len / cell_sz;
    if (len % cell_sz != 0) {
        IOD_DIE_ON_ERROR("transfer_size % iod_cell_size != 0",-1);
    }
    //IDEBUG(s->myrank,"Off %ld is cell %ld when cellsize is %ld",off,which_cell,len);
    s->slab->start[0]  = which_cell;
    s->slab->count[0]  = 1;
    s->slab->stride[0] = num_cells;
    s->slab->block[0]  = num_cells;
    setup_mem_desc(s, buf, len);
}

int
iod_num_cksums(iod_obj_type_t otype) {
    /* KV objs use 2 checksums per IO whereas blob/array only one */
    return (otype == IOD_OBJ_KV ? 2 : 1);
}

int
setup_cksum(iod_state_t *s, char *buf, size_t len, int rw, off_t off) {
    int ncksums = iod_num_cksums(s->otype);

    /* setup the checksum */
    if (s->checksum) {
        IDEBUG(rank, "Setting up checksums\n");
        if (rw==WRITE_MODE) {
            plfs_error_t pret;
            switch(s->otype) {
            case IOD_OBJ_BLOB:
            case IOD_OBJ_ARRAY:
                pret  = plfs_get_checksum(buf,len,(Plfs_checksum*)s->cksum);
                IOD_RETURN_ON_ERROR("plfs_get_checksum", pret);
                break;
            case IOD_OBJ_KV:
                pret  = plfs_get_checksum((char*)&off,sizeof(off_t),
                    (Plfs_checksum*)&(s->cksum[0]));
                IOD_RETURN_ON_ERROR("plfs_get_checksum", pret);
                pret  = plfs_get_checksum(buf,len,(Plfs_checksum*)&(s->cksum[1]));
                IOD_RETURN_ON_ERROR("plfs_get_checksum", pret);
                break;
            default:
                assert(0);
                break;
            }
        } else {
                memset(s->cksum,0,sizeof(iod_checksum_t) * ncksums);
        }
    }
    return 0;
}

// returns zero on success and number of bytes transferred in bytes
int 
iod_write(iod_state_t *I,char *buf,size_t len,off_t off,size_t cell_size, ssize_t *bytes) {
	iod_ret_t ret;
	iod_hint_list_t *hints = NULL;
	iod_event_t *event = NULL;
        //iod_checksum_t cs[2];
        //cs[0] = cs[1] = NULL;

        ret = setup_cksum(I, buf, len, WRITE_MODE, off);
        IOD_RETURN_ON_ERROR("setup_cksum",ret); 

        switch(I->otype) {
        case IOD_OBJ_BLOB:
            setup_blob_io(len,off,buf,I);
            ret =iod_blob_write(I->oh,I->tid,hints,I->mem_desc,I->io_desc,I->cksum,event);
            IOD_RETURN_ON_ERROR("iod_blob_write",ret); // successful write returns zero
            *bytes = len;
            break;
        case IOD_OBJ_KV:
            setup_kv_io(len,&off,buf,I);
            ret = iod_kv_set(I->oh,I->tid,hints,&(I->kv),I->cksum,event);
            IOD_RETURN_ON_ERROR("iod_kv_set",ret); // successful write returns zero
            *bytes = len;
            break;
        case IOD_OBJ_ARRAY:
            setup_array_io(len,off,buf,I,cell_size);
            ret = iod_array_write(I->oh,I->tid,hints,I->mem_desc,I->slab,I->cksum,event);
            IOD_RETURN_ON_ERROR("iod_array_write",ret); // successful write returns zero
            *bytes = len;
            break;
        default:
            assert(0);
            break;
        }

	return ret;
}

int 
iod_read(iod_state_t *s, char *buf,size_t len,off_t off,size_t cell_size, ssize_t *bytes) {
    iod_hint_list_t *hints = NULL;
    iod_event_t *event = NULL;
    iod_ret_t ret = 0;

    ret = setup_cksum(s, buf, len, READ_MODE, off);
    IOD_RETURN_ON_ERROR("setup_cksum",ret); 

    switch(s->otype) {
    case IOD_OBJ_ARRAY:
        setup_array_io(len,off,buf,s,cell_size);
        ret = iod_array_read(s->oh,s->tag,hints,s->mem_desc,s->slab,s->cksum,event);
        IOD_RETURN_ON_ERROR("iod_array_read",ret); // successful read returns zero
        *bytes = len;
        break;
    case IOD_OBJ_BLOB:
        setup_blob_io(len,off,buf,s);
        ret=iod_blob_read(s->oh,s->tag,hints,s->mem_desc,s->io_desc,s->cksum,event);
        IOD_RETURN_ON_ERROR("iod_blob_read",ret); // successful read returns zero
        *bytes = len;
        break;
    case IOD_OBJ_KV:
        setup_kv_io(len,&off,buf,s);
        ret = iod_kv_get_value(s->oh, s->tag, s->kv.key, s->kv.key_len, 
                s->kv.value, &(s->kv.value_len), s->cksum, event);
        IOD_RETURN_ON_ERROR("iod_kv_get",ret); // successful write returns zero
        *bytes = len;
        break;
    default:
        assert(0);
        break;
    }

    if (s->checksum) {
        IDEBUG(rank, "Verifying checksums\n");
        plfs_error_t pret = 0;
        if (s->otype == IOD_OBJ_KV) {
            pret = plfs_checksum_match((char*)&off,sizeof(off_t),
                                (Plfs_checksum)(*(&(s->cksum[0]))));
            IOD_RETURN_ON_ERROR("plfs_checksum_match", pret);
            pret = plfs_checksum_match(buf,len,
                                (Plfs_checksum)(*(&(s->cksum[1]))));
            IOD_RETURN_ON_ERROR("plfs_checksum_match", pret);
        } else {
            pret = plfs_checksum_match(buf,len,(Plfs_checksum)(*s->cksum));
            IOD_RETURN_ON_ERROR("plfs_checksum_match", pret);
        }
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
        StartTimer();
        ret=iod_trans_start(s->coh, &(s->tid),NULL,0,IOD_TRANS_R,NULL);
        AddTimer("iod_start_trans_read");
        IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);
    }
    IOD_Barrier(s);

    /* now open for read */
    StartTimer();
    ret = iod_obj_open_read(s->coh, s->oid, s->tid, NULL, &(s->oh), NULL);
    AddTimer("iod_obj_open_read");
    IOD_RETURN_ON_ERROR("iod_obj_open_read", ret);

    /* do the fetch if requested */
    s->tag = s->tid; // use tag for read, set to tid in case we don't fetch
    if (param->open == READ && param->iod_fetch) {
        IOD_Barrier(s);
	if( !s->myrank || param->filePerProc == TRUE ) {
            StartTimer();
            ret = iod_fetch(s);
            AddTimerAndBandwidth("iod_fetch", param->expectedAggFileSize);
            IOD_DIE_ON_ERROR("iod_fetch",ret);
        }
        if (! param->filePerProc ) {
            MPI_Bcast(&(s->tag), 1, MPI_UINT64_T, 0, s->mcom);
        }
        IOD_Barrier(s);
    }


    return ret;
}

int
iod_set_otype( iod_state_t *s, const char *type ) {
    if (type == NULL) {
        s->otype = IOD_OBJ_BLOB; 
    } else if (strcmp(type,"blob")==0) {
        s->otype = IOD_OBJ_BLOB;
    } else if (strcmp(type,"kv")==0) {
        s->otype = IOD_OBJ_KV;
    } else {
        assert(strcmp(type,"array")==0);
        s->otype = IOD_OBJ_ARRAY;
    }
    IDEBUG(s->myrank, "OBJ Type is %s", type);
}

int
open_wr(iod_state_t *I, char *filename, IOR_param_t *param) {
	int ret = 0;

	/* start the write trans */
	I->tid++; // bump up the tid
	if (I->myrank == 0) {
		ret = iod_trans_start(I->coh, &(I->tid), NULL, 0, IOD_TRANS_W, NULL);
		IDEBUG(I->myrank, "iod_trans_start %d : %d", I->tid, ret );
		IOD_RETURN_ON_ERROR("iod_trans_start", ret);
	}
        StartTimer();
        IOD_Barrier(I);

	/* create the obj */
	/* for shared file, only rank 0 does create then a barrier */
	/* or does create, most fail with EEXIST, no barrier */

	if( !I->myrank || param->filePerProc == TRUE ) {
		iod_hint_list_t *hints = NULL;
		set_checksum(&hints,I->checksum);
		ret = create_obj(I->coh, I->tid, &(I->oid), I->otype, I->array, hints, 
                                I->myrank);
		if (hints) free(hints);
		if ( ret != 0 ) {
			return ret;
		} else {
			IDEBUG(I->myrank,"iod obj %lli created successfully.",I->oid);
		}
	}
	IOD_Barrier(I);
        AddTimer("iod_obj_create");

	/* now open the obj */
        StartTimer();
	ret = iod_obj_open_write(I->coh, I->oid, I->tid, NULL, &(I->oh), NULL);
	IOD_RETURN_ON_ERROR("iod_obj_open_write", ret);
        AddTimer("iod_obj_open_write");
	return ret;	
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

    /* get some initial stuff from the params */
    istate->mcom = param->testComm;
    MPI_Comm_rank(istate->mcom, &(istate->myrank));
    MPI_Comm_size(istate->mcom, &(istate->nranks));
    istate->checksum = param->iod_checksum;

    if (!param->iod_cellsize) param->iod_cellsize = param->transferSize;

    StartTimer();
    IDEBUG(rank,"About to init");
    rc = iod_initialize(istate->mcom, NULL, istate->nranks, istate->nranks);
    IDEBUG(rank,"Done with init");
    AddTimer("iod_initialize");
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    if (rc != 0) {
        free(istate);
        return NULL;
    }

    // set the oid now here.
    iod_set_otype(istate,param->iod_type);
    if (param->filePerProc == TRUE) {
        istate->oid = istate->myrank;
    } else {
        istate->oid = 0;
    }
    IOD_OBJID_SETOWNER_APP(istate->oid)
    IOD_OBJID_SETTYPE(istate->oid, istate->otype);

    /* setup the io and memory descriptors used for blob io */
    istate->mem_desc = 
            malloc(sizeof(iod_mem_desc_t) + sizeof(iod_mem_frag_t));
    istate->io_desc = 
            malloc(sizeof(iod_blob_iodesc_t) + sizeof(iod_blob_iofrag_t));
    assert(istate->mem_desc && istate->io_desc);

    /* setup the checksums */ 
    if (istate->checksum) {
        IDEBUG(istate->myrank, "Using checksums");
        istate->cksum = malloc(sizeof(iod_checksum_t) * iod_num_cksums(istate->otype));
        assert(istate->cksum);
    } else {
        IDEBUG(istate->myrank, "Not using checksums");
        istate->cksum = NULL;
    }

    /* setup the array */
    if (istate->otype == IOD_OBJ_ARRAY) {
        size_t cell_size, total_sz, last_cell;
        cell_size = param->iod_cellsize; 
        total_sz = param->expectedAggFileSize; 
        last_cell = (total_sz / cell_size);
        setup_array(istate, cell_size, last_cell);
    } else {
        istate->array = NULL;
    }

    if (istate->otype == IOD_OBJ_KV && param->transferSize > IOD_KV_VALUE_MAXLEN) {
        assert(param->transferSize <= IOD_KV_VALUE_MAXLEN);
        return NULL; //return EFBIG;
    }

    /* open the container */
    IOD_Barrier(istate);
    rc = ContainerOpen(param->testFileName, param, istate);
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    IOD_Barrier(istate);

    /* skip tid 0 */
    StartTimer();
    rc = SkipTidZero(istate,param->testFileName);
    DCHECK(rc, "%s:%d", __FILE__, __LINE__);
    AddTimer("iod_skip_tid0");
    IOD_Barrier(istate);

    return istate;
}

/*
 * Creat and open a file through the IOD interface.
 */
static void *IOD_Create(char *testFileName, IOR_param_t * param)
{
    return IOD_Open(testFileName, param);
}

static int IOD_Init(char *filename, IOR_param_t *param) {
    int rc;
    Init(param);
    return rc;
} 

static int IOD_Fini(char *filename, IOR_param_t *param) {
    int rc;
    StartTimer();
    IDEBUG(istate->myrank,"About to close container");
    rc = iod_container_close(istate->coh, NULL, NULL);
    IDEBUG(istate->myrank,"Closed container: %d", rc);
    IOD_Barrier(istate);
    AddTimer("iod_container_close");
    IOD_RETURN_ON_ERROR("iod_container_close",rc);
    StartTimer();
    rc = iod_finalize(NULL);
    IDEBUG(istate->myrank,"iod_finalize: %d", rc);
    AddTimer("iod_finalize");

    if(istate->myrank==0) {
        PrintTimers();
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
        rc = iod_write(s,(char*)buffer,length,param->offset, param->iod_cellsize,&bytes); 
    } else {
        rc = iod_read(s,(char*)buffer,length,param->offset, param->iod_cellsize,&bytes); 
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

int
iod_close( iod_state_t *s,IOR_param_t * param) {
    iod_ret_t ret;

    /* finish the transaction */
    if (s->myrank == 0) {
            ret = iod_trans_finish(s->coh, s->tid, NULL, 0, NULL);
            IOD_RETURN_ON_ERROR("iod_trans_finish",ret);
            IDEBUG(s->myrank,"iod_trans_finish %d: success", s->tid);
    }
    IOD_Barrier(s);

    /* persist if requested */
    if (param->open == WRITE && param->iod_persist) {
        StartTimer();
        ret = iod_persist(s);
        AddTimerAndBandwidth("iod_persist", param->expectedAggFileSize);
        IOD_DIE_ON_ERROR("iod_persist",ret);
    }

    /* purge if requested */
    if (param->open == WRITE && param->iod_purge) {
        if (!param->iod_persist) IOD_DIE_ON_ERROR("purge requires persist",-1);
        assert(param->iod_persist);
        IOD_Barrier(s);
	if( !s->myrank || param->filePerProc == TRUE ) {
            StartTimer();
            ret = iod_purge(s);
            AddTimer("iod_purge");
            IOD_DIE_ON_ERROR("iod_purge",ret);
        }
        IOD_Barrier(s);
    }

    /* close the object handle */
    ret = iod_obj_close(s->oh, NULL, NULL);
    IOD_RETURN_ON_ERROR("iod_obj_close",ret);
    
    return ret;
}

/* caller must call this with all ranks if file_per_proc else only 0 */
int
iod_purge(iod_state_t *s) {
    iod_ret_t ret = 0;
    IDEBUG(s->myrank,"Purge on %lli @ %d", s->oid, s->tid);
    ret = iod_obj_purge(s->oh,s->tid,NULL,NULL);
    return ret;
}

/* caller must call this with all ranks if file_per_proc else only 0 */
int
iod_fetch(iod_state_t *s) {
    /* XXX TODO: Do a good layout for read.  Especially if file per proc */
    iod_ret_t ret = 0;
    IDEBUG(s->myrank,"Fetch on %lli @ %d", s->oid, s->tid);
    /* note: should this be fetch or replica ? */
    ret = iod_obj_fetch(s->oh,s->tid,NULL,NULL,NULL,&(s->tag),NULL);
    return ret;
}

int
iod_persist(iod_state_t *s) {
    iod_ret_t ret = 0;
    IOD_Barrier(s);
    if (s->myrank == 0) {
            ret = iod_trans_start(s->coh, &(s->tid), NULL, 0, IOD_TRANS_R, NULL);
            IOD_RETURN_ON_ERROR("iod_trans_start", ret);

            IDEBUG(s->myrank,"Persist on TR %d", s->tid);
            ret = iod_trans_persist(s->coh, s->tid, NULL, NULL);
            IOD_RETURN_ON_ERROR("iod_trans_persist", ret);

            ret = iod_trans_finish(s->coh, s->tid, NULL, 0, NULL);
            IOD_RETURN_ON_ERROR("iod_trans_finish", ret);
    }
    IOD_Barrier(s);
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

    StartTimer();
    IOD_Barrier(s);
    IDEBUG(s->myrank,"About to close object");
    ret = iod_close(s,param);
    IOD_DIE_ON_ERROR("iod_object_close",ret);
    IDEBUG(s->myrank,"Closed object");
    IOD_Barrier(s);
    sprintf(func_name, "iod_obj_close_%s", param->open==WRITE?"write":"read");
    AddTimer(func_name);

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
