      PROGRAM HDF_READER

      USE HDF5

     !USE HDF5 ! This module contains all necessary modules

      IMPLICIT NONE

      CHARACTER(LEN=8), PARAMETER :: filename = "TF.hdf5" ! File name
      CHARACTER(LEN=4), PARAMETER :: dsetname = "FT"     ! Dataset name

      INTEGER(HID_T) :: file_id       ! File identifier
      INTEGER(HID_T) :: dset_id       ! Dataset identifier
      INTEGER(HID_T) :: space_id       ! Dataspace identifier
      INTEGER(HID_T) :: dtype_id       ! Dataspace identifier

      INTEGER     ::   error ! Error flag
      INTEGER     ::  i, j, cols, rows

      REAL(KIND = 8), DIMENSION(:,:), ALLOCATABLE :: dset_data
      INTEGER(HSIZE_T), DIMENSION(2) :: data_dims
      INTEGER(HSIZE_T), DIMENSION(2) :: max_dims                  


      print *, 'Starting HDF5 Fortran Read'



   ! Initialize FORTRAN interface.

      CALL h5open_f(error)


   ! Open an existing file.

      CALL h5fopen_f (filename, H5F_ACC_RDWR_F, file_id, error)


   ! Open an existing dataset.

      CALL h5dopen_f(file_id, dsetname, dset_id, error)


   !Get dataspace ID
      CALL h5dget_space_f(dset_id, space_id,error)


   !Get dataspace dims

      CALL h5sget_simple_extent_dims_f(space_id,data_dims, max_dims, error)

      cols = data_dims(1)
      rows = data_dims(2)

   !Allocate dimensions to dset_data for reading
      ALLOCATE(dset_data(cols, rows))


   !Get data
      CALL h5dread_f(dset_id, H5T_NATIVE_INTEGER, dset_data, data_dims, error)



      print *, dset_data

      CALL h5close_f(error)



      END PROGRAM HDF_READER