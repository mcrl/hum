/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2020 Seoul National University.                             */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 08826, Korea                           */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jaehoon Jung, Jungho Park, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

#include "HUMProgram.h"
#include "HUMContext.h"
#include "HUMDevice.h"
#include "HUMKernel.h"
#include "HUMEvent.h"
#include "HUMObject.h"
#include "HUMKernelInfo.h"
#include "Callbacks.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <pthread.h>
#include <stdint.h>


using namespace std;

CudaProgramBinary::CudaProgramBinary(const unsigned char* binary,
                                 size_t size) 
{
  type_ = HUM_PROGRAM_BINARY_TYPE_CUDA;
	binary_ = (unsigned char*)malloc(sizeof(unsigned char) * size);
	memcpy(binary_, binary, sizeof(unsigned char) * size);
	size_ = size;
}


CudaProgramBinary::CudaProgramBinary(const char* filename)
{
	type_ = HUM_PROGRAM_BINARY_TYPE_CUDA;
	binary_ = NULL;
	size_ = 0;

	ReadBinaryFile(filename, size_);
}

CudaProgramBinary::CudaProgramBinary()
{
	type_ = HUM_PROGRAM_BINARY_TYPE_CUDA;
	binary_ = NULL;
	size_ = 0;
}

CudaProgramBinary::~CudaProgramBinary()
{
	if(binary_) free(binary_);
}

void CudaProgramBinary::ReadBinaryFile(const char* filename, size_t& file_size)
{
	if(binary_) free(binary_);

	file_size = 0;
	FILE* fp = fopen(filename, "rb");
	if (fp) {
		fseek(fp, 0, SEEK_END);
		file_size = ftell(fp);

		binary_ = new unsigned char[file_size + 1];
		fseek(fp, 0, SEEK_SET);

		fread(binary_, sizeof(unsigned char), file_size, fp);
		fclose(fp);

		binary_[file_size] = '\0';


		HUM_DEV("Completion to open %s, size=%ld", filename, file_size);
	}
	else {
		HUM_DEV("Unable to open %s, skipping...", filename);
	}
}
