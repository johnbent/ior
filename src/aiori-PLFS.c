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
* Implement of abstract I/O interface for PLFS.
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
static void *PLFS_Create(char *, IOR_param_t *);
static void *PLFS_Open(char *, IOR_param_t *);
static IOR_offset_t PLFS_Xfer(int, void *, IOR_size_t *,
                   IOR_offset_t, IOR_param_t *);
static void PLFS_Close(void *, IOR_param_t *);
static void PLFS_Delete(char *, IOR_param_t *);
static void PLFS_SetVersion(IOR_param_t *);
static void PLFS_Fsync(void *, IOR_param_t *);
static IOR_offset_t PLFS_GetFileSize(IOR_param_t *, MPI_Comm, char *);
static int PLFS_Init(char *, IOR_param_t *);
static int PLFS_Fini(char *, IOR_param_t *);


/************************** G L O B A L   V A R S *****************************/
MPI_Comm mcom;


/************************** D E C L A R A T I O N S ***************************/

ior_aiori_t plfs_aiori = {
    "PLFS",
    PLFS_Create,
    PLFS_Open,
    PLFS_Xfer,
    PLFS_Close,
    PLFS_Delete,
    PLFS_SetVersion,
    PLFS_Fsync,
    PLFS_GetFileSize,
    PLFS_Init,
    PLFS_Fini
};

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
pdebug_on(int rank) {
    switch(verbosity_level) {
            case DEBUG_ZERO: return (rank == 0);
            case DEBUG_NONE: return 0;
            case DEBUG_ALL: return (rank != -1);
            case DEBUG_EVERY: return 1;
    }
}

#define PDEBUG(rank, format, ...)                     \
do {                                    \
    int _rank = (rank);                         \
                                    \
    if (pdebug_on(_rank)) {                          \
        fprintf(stderr, "%.2f PLFS DEBUG (%s:%d): %d: : "       \
            format"\n", MPI_Wtime(), \
            __FILE__, __LINE__, rank,       \
            ##__VA_ARGS__);         \
        fflush(stdout);                     \
    }                                   \
} while (0);

#define PLFS_RETURN_ON_ERROR(X,Y) { \
	if (Y != 0 ) { \
		PLFS_PRINT_ERR(X,Y);\
		return Y; \
	} \
}

#define PLFS_DIE_ON_ERROR(X,Y) { \
	if (Y != 0 ) { \
		PLFS_PRINT_ERR(X,Y);\
		assert(0); \
	} \
}

#define PLFS_PRINT_ERR(X,Y) { \
	fprintf(stderr,"PLFS Error in %s:%d on %s: %s\n", \
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

static void PLFS_Barrier() {
    PDEBUG(rank, "MPI_Barrier");
    MPI_Barrier(mcom);
}

/*
 * Creat and open a file through the PLFS interface.
 */
static void *PLFS_Create(char *testFileName, IOR_param_t * param)
{
    return PLFS_Open(testFileName, param);
}

static int PLFS_Init(char *filename, IOR_param_t *param) {
    int rc;

    switch(param->verbose) {
    case 0: verbosity_level = DEBUG_NONE; break; 
    case 1: verbosity_level = DEBUG_ZERO; break;
    case 2: verbosity_level = DEBUG_ALL; break;
    default: verbosity_level = DEBUG_EVERY; break;
    } 

    mcom = param->testComm;
    
    PDEBUG(rank, "PLFS Init all done");

    return rc;
} 

static int PLFS_Fini(char *filename, IOR_param_t *param) {
    int rc;
    if(rank==0) {
        PrintTimers();
    }
    return rc;
}


/*
 * Open a file through the PLFS interface.
 * Opens both the container in RDWR and the object 
 * in whatever mode is necessary
 */
static void *PLFS_Open(char *testFileName, IOR_param_t * param)
{
    int rc;

    Plfs_fd *pfd = NULL;
    plfs_error_t pret;
    mode_t mode = 0666;
    int flags = 0;
    char *timer_type = NULL;

    if (param->open == WRITE) {
        flags = O_CREAT | O_WRONLY;
        if (!rank) flags |= O_TRUNC; // only 0 truncs
        timer_type = "plfs_open_write";
    } else {
        assert(param->open == READ);
        flags = O_RDONLY;
        timer_type = "plfs_open_read";
    }
        
    StartTimer();
    pret = plfs_open(&pfd, testFileName, flags, (pid_t)rank, mode, NULL);
    AddTimer(timer_type);
    PDEBUG(rank,"plfs_open %s: %d", testFileName, pret);
    if (pret == 0) {
        return (void*)pfd;
    } else {
        return NULL;
    }
}

/*
 * Write or read access to file using the PLFS interface.
 */
static IOR_offset_t PLFS_Xfer(int access, void *file, IOR_size_t * buffer,
                   IOR_offset_t length, IOR_param_t * param)
{
    plfs_error_t pret;
    ssize_t bytes = 0;
    Plfs_fd *pfd = (Plfs_fd*)file;  
    if (param->open == WRITE) {
        pret = plfs_write(pfd, (char*)buffer, length, param->offset, rank, &bytes);
    } else {
        assert(param->open == READ);
        pret = plfs_read(pfd, (char*)buffer, length, param->offset, &bytes);
    }
    return (pret == 0 ? bytes : -1);
}

/*
 * Perform fsync().
 */
static void PLFS_Fsync(void *fd, IOR_param_t * param)
{
    plfs_error_t pret;
    Plfs_fd *pfd = (Plfs_fd*)fd;
    pret = plfs_sync(pfd);
    PDEBUG(rank,"plfs_sync: %d",pret);
    PLFS_DIE_ON_ERROR("plfs_sync",pret);
}

/*
 * Close a file through the PLFS interface.
 */
static void PLFS_Close(void *fd, IOR_param_t * param)
{
    Plfs_fd *pfd = (Plfs_fd*)fd;
    plfs_error_t pret;
    int flags;
    int open_handles;
    uid_t uid = getuid();
    flags = ( param->open == READ ? O_RDONLY : O_CREAT | O_WRONLY );
    pret = plfs_close(pfd, rank, uid, flags, NULL, &open_handles);
    assert(open_handles==0);
    PDEBUG(rank,"plfs_close: %d",pret);
    PLFS_DIE_ON_ERROR("plfs_close",pret);
}

/*
 * Delete a file through the PLFS interface.
 */
static void PLFS_Delete(char *testFileName, IOR_param_t * param)
{
    StartTimer();
    plfs_error_t pret = plfs_unlink(testFileName);
    PDEBUG(rank,"plfs_unlink: %d",pret);
    PLFS_DIE_ON_ERROR("plfs_unlink",pret);
    AddTimer("plfs_unlink");
}

/*
 * Determine api version.
 */
static void PLFS_SetVersion(IOR_param_t * test)
{
    strcpy(test->apiVersion, test->api);
}

/*
 * Use PLFS stat() to return aggregate file size of all objects moved
 */
static IOR_offset_t PLFS_GetFileSize(IOR_param_t * test, MPI_Comm testComm,
                      char *testFileName)
{
    struct stat stat_buf;
    IOR_offset_t aggFileSizeFromStat, tmpMin, tmpMax, tmpSum;

    plfs_error_t pret;
    pret = plfs_getattr(NULL, testFileName, &stat_buf, 1); 
    PDEBUG(rank,"plfs_getattr %s : %lld\n", testFileName, stat_buf.st_size);

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
