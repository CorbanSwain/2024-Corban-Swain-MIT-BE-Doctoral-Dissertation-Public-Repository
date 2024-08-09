//
// An example of how to read a PixeLINK Data Stream (PDS) file and process the 
// images within. 
//
// The format of the PSD file is described in the API documentation on the page "PixeLINK Data Stream Format".
// 
// For this demo, we extract the individual images and generate a BMP file for each one.
//
// Pass the name of the PDS file on the command line.
//
#include <stdio.h>
#include <PixeLINKApi.h>
#include <PixeLINKTypes.h>
#include <vector>

#include <assert.h>
#include <iostream>
#define ASSERT(x) do { assert((x)); } while(0)

enum {
	PARAM_EXE_NAME = 0,
	PARAM_FRAME_COMMAND,
	PARAM_PDS_NAME,
	MIN_NUM_PARAMS
};

enum FrameCommand {
	COMMAND_INVALID,
	COMMAND_CREATE_BMPS,
	COMMAND_CREATE_CSV,
	COMMAND_PRINT_FRAME_INFO,
	COMMAND_CREATE_AVI
};

// Local function prototypes
static void				PrintUsage();
static float			GetPixelSize(U32 pixelFormat);
static PXL_RETURN_CODE	ProcessPdsFile(const char* pFileName, FrameCommand command);
static PXL_RETURN_CODE	ProcessPdsFileImpl(FILE* hFile, FrameCommand command);
static PXL_RETURN_CODE	ProcessFrame(void const* pFrame, FRAME_DESC const* pFrameDesc, FrameCommand command);
static PXL_RETURN_CODE	CreateBmp(void const* pFrame, FRAME_DESC const* pFrameDesc);
static PXL_RETURN_CODE	CreateCsv(void const* pFrame, FRAME_DESC const* pFrameDesc);
static PXL_RETURN_CODE	PrintFrameInfo(void const* pFrame, FRAME_DESC const* pFrameDesc);
static PXL_RETURN_CODE	SaveImageToFile(const char* pFilename, const char* pImage, const U32 imageSize);



static void
PrintUsage()
{
	printf("Usage: pdsreader <-bmp|-csv|-frameinfo> <name of pds file>\n");
	printf("Examples:\n");
	printf("\n");
	printf("Extract images in the PDS and create a BMP file:\n");
	printf("  pdsreader -bmp somefile.pds\n");
	printf("\n");
	printf("Extract images in the PDS and spit out the pixels as comma-separated values\n");
	printf("  pdsreader -csv somefile.pds\n");
	printf("\n");
	printf("Print the frame information  of the captured images\n");
	printf("  pdsreader -frameinfo somefile.pds\n");
	printf("\n");
	printf("Convert the PDS to an avi file\n");
	printf("  pdsreader -avi somefile.pds\n");
	printf("\n");
}

int
main(int argc, char* argv[])
{
	if (argc < MIN_NUM_PARAMS) {
		PrintUsage();
		return EXIT_FAILURE;
	}

	// What frame command are we supposed to do?
	FrameCommand command = COMMAND_INVALID;
	if (0 == strcmp("-bmp", argv[PARAM_FRAME_COMMAND])) {
		command = COMMAND_CREATE_BMPS;
	}
	else if (0 == strcmp("-csv", argv[PARAM_FRAME_COMMAND])) {
		command = COMMAND_CREATE_CSV;
	}
	else if (0 == strcmp("-frameinfo", argv[PARAM_FRAME_COMMAND])) {
		command = COMMAND_PRINT_FRAME_INFO;
	}
	else if (0 == strcmp("--avi", argv[PARAM_FRAME_COMMAND])) {
		command = COMMAND_CREATE_AVI;
		printf("Creating an avi file...\n");
	}
	else {
		printf("ERROR: The frame command '%s' is unrecognised\n", argv[PARAM_FRAME_COMMAND]);
		PrintUsage();
		return EXIT_FAILURE;
	}

	PXL_RETURN_CODE rc = ProcessPdsFile(argv[PARAM_PDS_NAME], command);
	if (!API_SUCCESS(rc)) {
		printf("An error occurred processing \"%s\" (0x%8.8X)\n", argv[PARAM_PDS_NAME], rc);
		PrintUsage();
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

//
// Open the PDS file, if possible, then pass the file handle off for processing of the file.
//
static PXL_RETURN_CODE
ProcessPdsFile(char const* const pFileName, const FrameCommand frameCommand)
{
	ASSERT(NULL != pFileName);
	PXL_RETURN_CODE rc;
	if (frameCommand == COMMAND_CREATE_AVI) {
		char outputFileName[MAX_PATH] = "";
		sprintf(&outputFileName[0], "%s.avi", pFileName);
		printf("outputFileName: \"%s\"\n", outputFileName);
		printf("pFileName:      \"%s\"\n", pFileName);

		rc = PxLFormatClip(pFileName, outputFileName, CLIP_FORMAT_AVI);
	}
	else {
		FILE* hFile = fopen(pFileName, "rb"); // open for binary read
		if (NULL == hFile) {
			return ApiInvalidParameterError;
		}

		printf("Reading %s... \n", pFileName);
		rc = ProcessPdsFileImpl(hFile, frameCommand);
		printf("done.\n");

		fclose(hFile);
	}

	return rc;
}

//
// We have a couple assumptions about the PDS file:
//     1) all the frame descriptors are the same size.
//     2) all the images are the same size.
//
static PXL_RETURN_CODE
ProcessPdsFileImpl(FILE* const hFile, const FrameCommand frameCommand)
{
	ASSERT(NULL != hFile);

	// Read the first 4 bytes, to ensure that this PDS starts with the PSD 'magic number'.
	U32 magicNumber = 0;
	size_t numItemsRead = fread((void*)&magicNumber, sizeof(magicNumber), 1, hFile);
	if (numItemsRead != 1) return ApiIOError;
	if (PIXELINK_DATA_STREAM_MAGIC_NUMBER != magicNumber) return ApiInvalidParameterError;

	// Next field: number of frames in the PDS file.
	// For each frame there will be a FRAME_DESC, followed by the frame data.
	U32 numberOfFramesInFile = 0;
	numItemsRead = fread((void*)&numberOfFramesInFile, sizeof(numberOfFramesInFile), 1, hFile);
	if (numItemsRead != 1) return ApiIOError;
	if (0 == numberOfFramesInFile) return ApiSuccess; // Check for special case

	// Record the offset to the first FRAME_DESC so we can reset to this point later on.
	const long int OFFSET_FIRST_FRAME_DESC = ftell(hFile);
	if (-1L == OFFSET_FIRST_FRAME_DESC) return ApiIOError;

	// Read the size of each of the FRAME_DESCs by reading the first field from the first FRAME_DESC.
	// (The first field of a FRAME_DESC is its uSize field)
	U32 sizeOfFrameDesc = 0;
	numItemsRead = fread((void*)&sizeOfFrameDesc, sizeof(sizeOfFrameDesc), 1, hFile);
	if (numItemsRead != 1) return ApiIOError;
	ASSERT(sizeOfFrameDesc > 0);

	// Now that we know how big a FRAME_DESC is, rewind to the start of the first FRAME_DESC and
	// then read it in so we can use the information in it to determine how big a frame is.
	std::vector<U8> frameDescBuffer(sizeOfFrameDesc);
	FRAME_DESC* pFrameDesc = (FRAME_DESC*)&frameDescBuffer[0];	// A pointer of convenience.
	int seekResult = fseek(hFile, OFFSET_FIRST_FRAME_DESC, SEEK_SET);
	if (0 != seekResult) return ApiIOError;
	numItemsRead = fread((void*)&frameDescBuffer[0], frameDescBuffer.size(), 1, hFile);
	if (numItemsRead != 1) return ApiIOError;

	// Calculate the size of frame, and allocate a buffer big enough to hold one.
	const U32 numPixels = ((U32)pFrameDesc->Roi.fHeight / (U32)pFrameDesc->Decimation.fValue) *
		((U32)pFrameDesc->Roi.fWidth / (U32)pFrameDesc->Decimation.fValue);
	const float pixelSize = GetPixelSize((U32)pFrameDesc->PixelFormat.fValue);
	const U32 sizeOfFrame = (U32)((float)numPixels * pixelSize);
	if (0 == sizeOfFrame) return ApiIOError;
	std::vector<U8> frameBuffer(sizeOfFrame);

	// Move the file pointer back to the beginning of the first FRAME_DESC.
	// This makes our loop below simpler.
	seekResult = fseek(hFile, OFFSET_FIRST_FRAME_DESC, SEEK_SET);
	if (0 != seekResult) return ApiIOError;

	// And now, finally, read and process each of the images
	for (U32 i = 0; i < numberOfFramesInFile; i++) {

		printf("Processing image %d of %d... ", i, numberOfFramesInFile);
		// Read in the frame desc
		numItemsRead = fread((void*)&frameDescBuffer[0], frameDescBuffer.size(), 1, hFile);
		if (numItemsRead != 1) return ApiIOError;

		// Now read in the frame data
		numItemsRead = fread((void*)&frameBuffer[0], frameBuffer.size(), 1, hFile);
		if (numItemsRead != 1) return ApiIOError;

		// At this point, you can do whatever you want with the frame.
		// For this example, we'll convert it to a BMP.
		PXL_RETURN_CODE rc = ProcessFrame(&frameBuffer[0], pFrameDesc, frameCommand);
		if (!API_SUCCESS(rc)) return rc;

		printf("done.\n", i);
	}

	return ApiSuccess;
}


//
// Given the pixel format, return the size of a individual pixel (in bytes)
// Taken from the getsnapshot demo app (getsnapshot.c).
//
// Returns 0 on failure.
//
static float
GetPixelSize(const U32 pixelFormat)
{
	U32 retVal = 0;

	switch (pixelFormat)
	{
	case PIXEL_FORMAT_MONO8:
	case PIXEL_FORMAT_BAYER8_BGGR:
	case PIXEL_FORMAT_BAYER8_GBRG:
	case PIXEL_FORMAT_BAYER8_GRBG:
	case PIXEL_FORMAT_BAYER8_RGGB:
		return 1.0f;
	case PIXEL_FORMAT_MONO16:
	case PIXEL_FORMAT_BAYER16_BGGR:
	case PIXEL_FORMAT_BAYER16_GBRG:
	case PIXEL_FORMAT_BAYER16_GRBG:
	case PIXEL_FORMAT_BAYER16_RGGB:
	case PIXEL_FORMAT_YUV422:
		return 2.0f;
	case PIXEL_FORMAT_RGB24_DIB:
	case PIXEL_FORMAT_RGB24_NON_DIB:
	case PIXEL_FORMAT_BGR24_NON_DIB:
		return 3.0f;
	case PIXEL_FORMAT_RGBA:
	case PIXEL_FORMAT_BGRA:
	case PIXEL_FORMAT_ARGB:
	case PIXEL_FORMAT_ABGR:
		return 4.0f;
	case PIXEL_FORMAT_RGB48:
		return 6.0f;
	case PIXEL_FORMAT_MONO12_PACKED:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
	case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
		return 1.5f;
	case PIXEL_FORMAT_MONO10_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST:
		return 1.25f;
	case PIXEL_FORMAT_STOKES4_12:
	case PIXEL_FORMAT_POLAR4_12:
	case PIXEL_FORMAT_POLAR_RAW4_12:
	case PIXEL_FORMAT_HSV4_12:
		return 6.0f;

	}
	ASSERT(FALSE);
	return retVal;
}


//
// For demonstration purposes, we will take a frame and FRAME_DESC from the PDS file and
// convert it to a BMP, then save it as a BMP file. 
//
static PXL_RETURN_CODE
ProcessFrame(void const* const pFrame, FRAME_DESC const* const pFrameDesc, const FrameCommand frameCommand)
{
	switch (frameCommand)
	{
	case COMMAND_CREATE_BMPS:		return CreateBmp(pFrame, pFrameDesc);
	case COMMAND_CREATE_CSV:		return CreateCsv(pFrame, pFrameDesc);
	case COMMAND_PRINT_FRAME_INFO:	return PrintFrameInfo(pFrame, pFrameDesc);
	}

	ASSERT(0);
	return ApiInvalidParameterError;
}

static PXL_RETURN_CODE
CreateBmp(void const* const pFrame, FRAME_DESC const* const pFrameDesc)
{
	ASSERT(NULL != pFrame);
	ASSERT(NULL != pFrameDesc);

	// How big will the BMP be?
	U32 bmpSize = 0;
	PXL_RETURN_CODE rc = PxLFormatImage(pFrame, pFrameDesc, IMAGE_FORMAT_BMP, NULL, &bmpSize);
	if (!API_SUCCESS(rc)) return rc;

	// Allocate a buffer to hold the BMP
	std::vector<U8> bmpBuffer(bmpSize);
	// And now convert to a BMP
	rc = PxLFormatImage(pFrame, pFrameDesc, IMAGE_FORMAT_BMP, (void*)&bmpBuffer[0], &bmpSize);
	if (!API_SUCCESS(rc)) return rc;

	// Save the BMP to file, using the frame number as part of the file name.
	char fileName[MAX_PATH];
	sprintf(&fileName[0], "image%6.6d.bmp", pFrameDesc->uFrameNumber);
	rc = SaveImageToFile(&fileName[0], (char*)&bmpBuffer[0], (U32)bmpBuffer.size());

	return rc;
}

//
// Save a buffer to a file.
// This overwrites any existing file.
//
static PXL_RETURN_CODE
SaveImageToFile(char const* const pFilename, char const* const pImage, const U32 imageSize)
{
	ASSERT(NULL != pFilename);
	ASSERT(NULL != pImage);
	ASSERT(imageSize > 0);

	// Open our file for binary write
	FILE* pFile = fopen(pFilename, "wb");
	if (NULL == pFile) {
		return ApiIOError;
	}

	const size_t numItemsWritten = fwrite((void*)pImage, imageSize, 1, pFile);

	fclose(pFile);

	return (1 == numItemsWritten) ? ApiSuccess : ApiIOError;

}

//
// We just spit out the raw frame data on a byte-by-byte basis.
// Each row of the csv corresponds to a row in an individual image.
//
static PXL_RETURN_CODE
CreateCsv(void const* const pFrame, FRAME_DESC const* const pFrameDesc)
{
	ASSERT(NULL != pFrame);
	ASSERT(NULL != pFrameDesc);

	const U32 frameHeight = static_cast<U32>(pFrameDesc->Roi.fHeight / pFrameDesc->Decimation.fValue);
	const U32 frameWidth = static_cast<U32>(pFrameDesc->Roi.fWidth / pFrameDesc->Decimation.fValue);
	const float pixelSize = GetPixelSize(static_cast<U32>(pFrameDesc->PixelFormat.fValue));

	// Print the frame number
	printf("\nframe,%d\n", pFrameDesc->uFrameNumber);

	U8 const* pByte = reinterpret_cast<U8 const*>(pFrame);

	for (U32 r = 0; r < frameHeight; r++) {
		for (U32 c = 0; c < (U32)((float)frameWidth * pixelSize); c++) {
			printf("%d,", *pByte);
			pByte++;
		}
		printf("\n");
	}

	return ApiSuccess;
}


static PXL_RETURN_CODE
PrintFrameInfo(void const* const pFrame, FRAME_DESC const* const pFrameDesc)
{
	printf("\nFrame# %d, time %f, temp %5.2f\n", pFrameDesc->uFrameNumber, pFrameDesc->fFrameTime, pFrameDesc->Temperature.fValue);
	return ApiSuccess;
}
