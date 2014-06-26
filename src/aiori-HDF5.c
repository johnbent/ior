/* -*- mode: c; c-basic-offset: 8; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=8:tabstop=8:
 */
/******************************************************************************\
*                                                                              *
*        Copyright (c) 2003, The Regents of the University of California       *
*      See the file COPYRIGHT for a complete copyright notice and license.     *
*                                                                              *
********************************************************************************
*
*  Implement abstract I/O interface for HDF5.
*
\******************************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <stdio.h>              /* only for fprintf() */
#include <stdlib.h>
#include <sys/stat.h>
#include <hdf5.h>
#include <mpi.h>

#include "aiori.h"              /* abstract IOR interface */
#include "utilities.h"
#include "iordef.h"

#define NUM_DIMS 1              /* number of dimensions to data set */

/******************************************************************************/
/*
 * HDF5_CHECK will display a custom error message and then exit the program
 */

/*
 * should use MPI_Abort(), not exit(), in this macro; some versions of
 * MPI, however, hang with HDF5 property lists et al. left unclosed
 */

/*
 * for versions later than hdf5-1.6, the H5Eget_[major|minor]() functions
 * have been deprecated and replaced with H5Eget_msg()
 */
#if H5_VERS_MAJOR > 1 && H5_VERS_MINOR > 6
#define HDF5_CHECK(HDF5_RETURN, MSG) do {                                \
    char   resultString[1024];                                           \
                                                                         \
    if (HDF5_RETURN < 0) {                                               \
        fprintf(stdout, "** error **\n");                                \
        fprintf(stdout, "ERROR in %s (line %d): %s.\n",                  \
                __FILE__, __LINE__, MSG);                                \
        strcpy(resultString, H5Eget_major((H5E_major_t)HDF5_RETURN));    \
        if (strcmp(resultString, "Invalid major error number") != 0)     \
            fprintf(stdout, "HDF5 %s\n", resultString);                  \
        strcpy(resultString, H5Eget_minor((H5E_minor_t)HDF5_RETURN));    \
        if (strcmp(resultString, "Invalid minor error number") != 0)     \
            fprintf(stdout, "%s\n", resultString);                       \
        fprintf(stdout, "** exiting **\n");                              \
        exit(-1);                                                        \
    }                                                                    \
} while(0)
#else                           /* ! (H5_VERS_MAJOR > 1 && H5_VERS_MINOR > 6) */
#define HDF5_CHECK(HDF5_RETURN, MSG) do {                                \
    char   resultString[1024];                                           \
                                                                         \
    if (HDF5_RETURN < 0) {                                               \
        fprintf(stdout, "** error **\n");                                \
        fprintf(stdout, "ERROR in %s (line %d): %s.\n",                  \
                __FILE__, __LINE__, MSG);                                \
        /*                                                               \
         * H5Eget_msg(hid_t mesg_id, H5E_type_t* mesg_type,              \
         *            char* mesg, size_t size)                           \
         */                                                              \
        fprintf(stdout, "** exiting **\n");                              \
        exit(-1);                                                        \
    }                                                                    \
} while(0)
#endif                          /* H5_VERS_MAJOR > 1 && H5_VERS_MINOR > 6 */
/**************************** P R O T O T Y P E S *****************************/

static IOR_offset_t SeekOffset(void *, IOR_offset_t, IOR_param_t *);
static void SetupDataSet(void *, IOR_param_t *);
static void *HDF5_Create(char *, IOR_param_t *);
static void *HDF5_Open(char *, IOR_param_t *);
static IOR_offset_t HDF5_Xfer(int, void *, IOR_size_t *,
                           IOR_offset_t, IOR_param_t *);
static void HDF5_Close(void *, IOR_param_t *);
static void HDF5_Delete(char *, IOR_param_t *);
static void HDF5_SetVersion(IOR_param_t *);
static void HDF5_Fsync(void *, IOR_param_t *);
static IOR_offset_t HDF5_GetFileSize(IOR_param_t *, MPI_Comm, char *);
static int HDF5_Init(char *, IOR_param_t *);
static int HDF5_Fini(char *, IOR_param_t *);

/************************** D E C L A R A T I O N S ***************************/

ior_aiori_t hdf5_aiori = {
        "HDF5",
        HDF5_Create,
        HDF5_Open,
        HDF5_Xfer,
        HDF5_Close,
        HDF5_Delete,
        HDF5_SetVersion,
        HDF5_Fsync,
        HDF5_GetFileSize,
        HDF5_Init,
        HDF5_Fini
};

static hid_t xferPropList;      /* xfer property list */
hid_t dataSet;                  /* data set id */
hid_t dataSpace;                /* data space id */
hid_t fileDataSpace;            /* file data space id */
hid_t memDataSpace;             /* memory data space id */
int newlyOpenedFile;            /* newly opened file */

uint64_t version;
uint64_t trans_num;

int my_rank, my_size;
hid_t rid, tid;

/***************************** F U N C T I O N S ******************************/

static int HDF5_Init(char *filename, IOR_param_t *param) {
    int rc = 0;

    rc = EFF_init( MPI_COMM_WORLD, MPI_INFO_NULL );

    return rc;
} 

static int HDF5_Fini(char *filename, IOR_param_t *param) {
    int rc = 0;

    MPI_Barrier (MPI_COMM_WORLD);
    rc = EFF_finalize();
    MPI_Barrier (MPI_COMM_WORLD);

    return rc;
}

/*
 * Create and open a file through the HDF5 interface.
 */
static void *HDF5_Create(char *testFileName, IOR_param_t * param)
{
        return HDF5_Open(testFileName, param);
}

/*
 * Open a file through the HDF5 interface.
 */
static void *HDF5_Open(char *testFileName, IOR_param_t * param)
{
        hid_t accessPropList, createPropList;
        hsize_t memStart[NUM_DIMS],
            dataSetDims[NUM_DIMS],
            memStride[NUM_DIMS],
            memCount[NUM_DIMS], memBlock[NUM_DIMS], memDataSpaceDims[NUM_DIMS];
        int tasksPerDataSet;
        unsigned fd_mode = (unsigned)0;
        hid_t *fd;
        MPI_Comm comm;
        MPI_Info mpiHints = MPI_INFO_NULL;

        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &my_size);

        fd = (hid_t *) malloc(sizeof(hid_t));
        if (fd == NULL)
                ERR("malloc() failed");
        /*
         * HDF5 uses different flags than those for POSIX/MPIIO
         */
        if (param->open == WRITE) {     /* WRITE flags */
                param->openFlags = IOR_TRUNC;
        } else {                /* READ or check WRITE/READ flags */
                param->openFlags = IOR_RDONLY;
        }

        /* set IOR file flags to HDF5 flags */
        /* -- file open flags -- */
        if (param->openFlags & IOR_RDONLY) {
                fd_mode |= H5F_ACC_RDONLY;
        }
        if (param->openFlags & IOR_WRONLY) {
                fprintf(stdout, "File write only not implemented in HDF5\n");
        }
        if (param->openFlags & IOR_RDWR) {
                fd_mode |= H5F_ACC_RDWR;
        }
        if (param->openFlags & IOR_APPEND) {
                fprintf(stdout, "File append not implemented in HDF5\n");
        }
        if (param->openFlags & IOR_CREAT) {
                fd_mode |= H5F_ACC_CREAT;
        }
        if (param->openFlags & IOR_EXCL) {
                fd_mode |= H5F_ACC_EXCL;
        }
        if (param->openFlags & IOR_TRUNC) {
                fd_mode |= H5F_ACC_TRUNC;
        }
        if (param->openFlags & IOR_DIRECT) {
                fprintf(stdout, "O_DIRECT not implemented in HDF5\n");
        }

        /* set up file access property list */
        accessPropList = H5Pcreate(H5P_FILE_ACCESS);
        HDF5_CHECK(accessPropList, "cannot create file access property list");

        H5Pset_fapl_iod(accessPropList, MPI_COMM_WORLD, MPI_INFO_NULL);
        H5Pset_metadata_integrity_scope(accessPropList, H5_CHECKSUM_NONE);

        /* open file */
        if (param->open == WRITE) {     /* WRITE */
                hid_t trspl_id;

                *fd = H5Fcreate_ff(testFileName, fd_mode,
                                   H5P_DEFAULT, accessPropList, H5_EVENT_STACK_NULL);
                HDF5_CHECK(*fd, "cannot create file");

                if(0 == my_rank) {
                    version = 1;
                    rid = H5RCacquire(*fd, &version, H5P_DEFAULT, H5_EVENT_STACK_NULL);
                    HDF5_CHECK(rid, "cannot acquire read context");
                }
                MPI_Bcast(&version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
                assert(1 == version);
                if (my_rank != 0)
                        rid = H5RCcreate(*fd, version);

                /* create transaction object */
                tid = H5TRcreate(*fd, rid, (uint64_t)2);
                trspl_id = H5Pcreate (H5P_TR_START);
                H5Pset_trspl_num_peers(trspl_id, my_size);
                H5TRstart(tid, trspl_id, H5_EVENT_STACK_NULL);
                H5Pclose(trspl_id);
        } else {                /* READ or CHECK */
                *fd = H5Fopen_ff(testFileName, fd_mode, accessPropList, 
                                 &rid, H5_EVENT_STACK_NULL);
                HDF5_CHECK(*fd, "cannot open file");
        }

        /* this is necessary for resetting various parameters
           needed for reopening and checking the file */
        newlyOpenedFile = TRUE;

        HDF5_CHECK(H5Pclose(accessPropList),
                   "cannot close access property list");

        /* create property list for serial/parallel access */
        xferPropList = H5Pcreate(H5P_DATASET_XFER);
        HDF5_CHECK(xferPropList, "cannot create transfer property list");

        /* set up memory data space for transfer */
        memStart[0] = (hsize_t) 0;
        memCount[0] = (hsize_t) 1;
        memStride[0] = (hsize_t) (param->transferSize / sizeof(IOR_size_t));
        memBlock[0] = (hsize_t) (param->transferSize / sizeof(IOR_size_t));
        memDataSpaceDims[0] = (hsize_t) param->transferSize;
        memDataSpace = H5Screate_simple(NUM_DIMS, memDataSpaceDims, NULL);
        HDF5_CHECK(memDataSpace, "cannot create simple memory data space");

        /* define hyperslab for memory data space */
        HDF5_CHECK(H5Sselect_hyperslab(memDataSpace, H5S_SELECT_SET,
                                       memStart, memStride, memCount,
                                       memBlock), "cannot create hyperslab");

        /* set up parameters for fpp or different dataset count */
        if (param->filePerProc) {
                tasksPerDataSet = 1;
        } else {
                if (param->individualDataSets) {
                        /* each task in segment has single data set */
                        tasksPerDataSet = 1;
                } else {
                        /* share single data set across all tasks in segment */
                        tasksPerDataSet = param->numTasks;
                }
        }
        dataSetDims[0] = (hsize_t) ((param->blockSize / sizeof(IOR_size_t))
                                    * tasksPerDataSet);

        /* create a simple data space containing information on size
           and shape of data set, and open it for access */
        dataSpace = H5Screate_simple(NUM_DIMS, dataSetDims, NULL);
        HDF5_CHECK(dataSpace, "cannot create simple data space");

        /* create new data set */
        SetupDataSet(fd, param);

        return (fd);
}

/*
 * Write or read access to file using the HDF5 interface.
 */
static IOR_offset_t HDF5_Xfer(int access, void *fd, IOR_size_t * buffer,
                              IOR_offset_t length, IOR_param_t * param)
{
        static int firstReadCheck = FALSE, startNewDataSet;
        IOR_offset_t segmentPosition, segmentSize;

        /*
         * this toggle is for the read check operation, which passes through
         * this function twice; note that this function will open a data set
         * only on the first read check and close only on the second
         */
        if (access == READCHECK) {
                if (firstReadCheck == TRUE) {
                        firstReadCheck = FALSE;
                } else {
                        firstReadCheck = TRUE;
                }
        }

        /* determine by offset if need to start new data set */
        if (param->filePerProc == TRUE) {
                segmentPosition = (IOR_offset_t) 0;
                segmentSize = param->blockSize;
        } else {
                segmentPosition =
                    (IOR_offset_t) ((rank + rankOffset) % param->numTasks)
                    * param->blockSize;
                segmentSize =
                    (IOR_offset_t) (param->numTasks) * param->blockSize;
        }

#if 0
        if ((IOR_offset_t) ((param->offset - segmentPosition) % segmentSize) ==
            0) {
                /*
                 * ordinarily start a new data set, unless this is the
                 * second pass through during a read check
                 */
                startNewDataSet = TRUE;
                if (access == READCHECK && firstReadCheck != TRUE) {
                        startNewDataSet = FALSE;
                }
        }


        /* create new data set */
        if (startNewDataSet == TRUE) {
                /* if just opened this file, no data set to close yet */
                if (newlyOpenedFile != TRUE) {
                        HDF5_CHECK(H5Dclose(dataSet), "cannot close data set");
                        HDF5_CHECK(H5Sclose(fileDataSpace),
                                   "cannot close file data space");
                }
                SetupDataSet(fd, param);
        }
#endif

        SeekOffset(fd, param->offset, param);

        /* this is necessary to reset variables for reaccessing file */
        startNewDataSet = FALSE;
        newlyOpenedFile = FALSE;

        H5Pset_rawdata_integrity_scope(xferPropList, H5_CHECKSUM_NONE);
        /* access the file */
        if (access == WRITE) {  /* WRITE */
                HDF5_CHECK(H5Dwrite_ff(dataSet, H5T_NATIVE_LLONG,
                                       memDataSpace, fileDataSpace,
                                       xferPropList, buffer, tid, H5_EVENT_STACK_NULL),
                           "cannot write to data set");
        } else {                /* READ or CHECK */
                HDF5_CHECK(H5Dread_ff(dataSet, H5T_NATIVE_LLONG,
                                      memDataSpace, fileDataSpace,
                                      xferPropList, buffer, rid, H5_EVENT_STACK_NULL),
                           "cannot read from data set");
        }
        return (length);
}

/*
 * Perform fsync().
 */
static void HDF5_Fsync(void *fd, IOR_param_t * param)
{
        ;
}

/*
 * Close a file through the HDF5 interface.
 */
static void HDF5_Close(void *fd, IOR_param_t * param)
{
        if ( param->open == WRITE ) {
                H5TRfinish( tid, H5P_DEFAULT, NULL, H5_EVENT_STACK_NULL );
                H5TRclose(tid);
        }

        if ( param->open == WRITE ) {
                if ( 0 == my_rank )
                        H5RCrelease( rid, H5_EVENT_STACK_NULL );
        }
        else 
                H5RCrelease( rid, H5_EVENT_STACK_NULL );

        H5RCclose(rid);
        if (param->fd_fppReadCheck == NULL) {
                HDF5_CHECK(H5Dclose(dataSet), "cannot close data set");
                HDF5_CHECK(H5Sclose(dataSpace), "cannot close data space");
                HDF5_CHECK(H5Sclose(fileDataSpace),
                           "cannot close file data space");
                HDF5_CHECK(H5Sclose(memDataSpace),
                           "cannot close memory data space");
                HDF5_CHECK(H5Pclose(xferPropList),
                           " cannot close transfer property list");
        }

        MPI_Barrier (MPI_COMM_WORLD);

        if (param->open == WRITE && param->iod_persist)
                HDF5_CHECK(H5Fclose_ff(*(hid_t *) fd, 1 , H5_EVENT_STACK_NULL), "cannot close file");
        else
                HDF5_CHECK(H5Fclose_ff(*(hid_t *) fd, 0 , H5_EVENT_STACK_NULL), "cannot close file");

        free(fd);

}

/*
 * Delete a file through the HDF5 interface.
 */
static void HDF5_Delete(char *testFileName, IOR_param_t * param)
{
        ;
}

/*
 * Determine api version.
 */
static void HDF5_SetVersion(IOR_param_t * test)
{
        unsigned major, minor, release;
        if (H5get_libversion(&major, &minor, &release) < 0) {
                WARN("cannot get HDF5 library version");
        } else {
                sprintf(test->apiVersion, "%s-%u.%u.%u",
                        test->api, major, minor, release);
        }
#ifndef H5_HAVE_PARALLEL
        strcat(test->apiVersion, " (Serial)");
#else                           /* H5_HAVE_PARALLEL */
        strcat(test->apiVersion, " (Parallel)");
#endif                          /* not H5_HAVE_PARALLEL */
}

/*
 * Seek to offset in file using the HDF5 interface and set up hyperslab.
 */
static IOR_offset_t SeekOffset(void *fd, IOR_offset_t offset,
                                            IOR_param_t * param)
{
        IOR_offset_t segmentSize;
        hsize_t hsStride[NUM_DIMS], hsCount[NUM_DIMS], hsBlock[NUM_DIMS];
        hsize_t hsStart[NUM_DIMS];

        if (param->filePerProc == TRUE) {
                segmentSize = (IOR_offset_t) param->blockSize;
        } else {
                segmentSize =
                    (IOR_offset_t) (param->numTasks) * param->blockSize;
        }

        /* create a hyperslab representing the file data space */
        if (param->individualDataSets) {
                /* start at zero offset if not */
                hsStart[0] = (hsize_t) ((offset % param->blockSize)
                                        / sizeof(IOR_size_t));
        } else {
                /* start at a unique offset if shared */
                hsStart[0] =
                    (hsize_t) ((offset % segmentSize) / sizeof(IOR_size_t));
        }
        hsCount[0] = (hsize_t) 1;
        hsStride[0] = (hsize_t) (param->transferSize / sizeof(IOR_size_t));
        hsBlock[0] = (hsize_t) (param->transferSize / sizeof(IOR_size_t));

        /* retrieve data space from data set for hyperslab */
        fileDataSpace = H5Dget_space(dataSet);
        HDF5_CHECK(fileDataSpace, "cannot get data space from data set");
        HDF5_CHECK(H5Sselect_hyperslab(fileDataSpace, H5S_SELECT_SET,
                                       hsStart, hsStride, hsCount, hsBlock),
                   "cannot select hyperslab");

        return (offset);
}

/*
 * Create HDF5 data set.
 */
static void SetupDataSet(void *fd, IOR_param_t * param)
{
        char dataSetName[MAX_STR];
        hid_t dataSetPropList;
        int dataSetID;
        static int dataSetSuffix = 0;

        /* may want to use an extendable dataset (H5S_UNLIMITED) someday */
        /* may want to use a chunked dataset (H5S_CHUNKED) someday */

        /* need to reset suffix counter if newly-opened file */
        if (newlyOpenedFile)
                dataSetSuffix = 0;

        /* may want to use individual access to each data set someday */
        if (param->individualDataSets) {
                dataSetID = (rank + rankOffset) % param->numTasks;
        } else {
                dataSetID = 0;
        }

        sprintf(dataSetName, "%s-%04d.%04d", "Dataset", dataSetID,
                dataSetSuffix++);

        if (param->open == WRITE) {     /* WRITE */
                void *dset_token = NULL;
                size_t token_size = 0;
                herr_t ret;

                if(0 == rank) {
                        hid_t dcpl_id;
                        dcpl_id = H5Pcreate( H5P_DATASET_CREATE );
                        H5Pset_ocpl_enable_checksum(dcpl_id, H5_CHECKSUM_NONE);

                        /* create data set */
                        dataSet = H5Dcreate_ff(*(hid_t *) fd, dataSetName, H5T_NATIVE_LLONG,
                                               dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, 
                                               tid, H5_EVENT_STACK_NULL);
                        HDF5_CHECK(dataSet, "cannot create data set");

                        ret = H5Oget_token(dataSet, NULL, &token_size);
                        assert(0 == ret);
                        dset_token = malloc(token_size);
                        ret = H5Oget_token(dataSet, dset_token, &token_size);
                        assert(0 == ret);

                        H5Pclose(dcpl_id);
                }

                MPI_Bcast(&token_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

                if(0 != rank) {
                        dset_token = malloc(token_size);
                }

                MPI_Bcast(dset_token, token_size, MPI_BYTE, 0, MPI_COMM_WORLD);

                if(0 != rank) {
                        dataSet = H5Oopen_by_token(dset_token, tid, H5_EVENT_STACK_NULL);
                }

                free(dset_token);
        }
        else {                /* READ or CHECK */
                dataSet = H5Dopen_ff(*(hid_t *) fd, dataSetName, 
                                     H5P_DEFAULT, rid, H5_EVENT_STACK_NULL);
                HDF5_CHECK(dataSet, "cannot create data set");
        }
}

/*
 * Use MPIIO call to get file size.
 */
static IOR_offset_t
HDF5_GetFileSize(IOR_param_t * test, MPI_Comm testComm, char *testFileName)
{
        ;
}
