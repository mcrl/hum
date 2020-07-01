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

#include "hum.h"
#include "HUMPlatform.h"
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "MemoryRegion.h"
#include "Device/NVLibs.h"
#include "ioctl.h"


int grank_ = -1;
int gsize_ = 0;
#ifdef USE_MEM_PREFETCH
int driver_fd_;
#endif
HUMComm* g_HUMComm = NULL;
NVLibs* g_nvlibs = NULL;

//MemoryRegion<HUMMem> _g_MemRegion_(MEM_REGION_SIZE);
MemoryRegion<HUMMem>* g_MemRegion_ = NULL;

static int init_cnt = 0;

class HUM_Runtime {
	public:
		HUM_Runtime() {
			init_cnt++;

      grank_ = 0;
      gsize_ = 1;

			NVLibs* nvlibs = new NVLibs();
			g_nvlibs = nvlibs;

			g_MemRegion_ = new MemoryRegion<HUMMem>(MEM_REGION_SIZE);	

#ifdef USE_MEM_PREFETCH
			driver_fd_ = open("/dev/hum", O_RDWR);
			if(driver_fd_ == -1) {
				HUM_ERROR("Failed to open hum device fd=%d", driver_fd_);
				exit(0);
			}

			hum_int err = nvlibs->cudaSetDevice(0);
			if(err != 0) {
				HUM_ERROR("cudaSetDevice(%d) Failed\n", 0);
				assert(0);
			}

			ioctl(driver_fd_, IOCTL_HOOK);
#endif

			run_ = 0;
			//run();

			HUM_DEV("HUM_Runtime %p start!", this);
		};

		~HUM_Runtime() {
			HUM_DEV("HUM_Runtime %p end!", this);


			if (rank_ == 0) {
#ifdef USE_MEM_PREFETCH
				close(driver_fd_);
#endif
				delete(g_MemRegion_);

				if(g_nvlibs) {
					delete(g_nvlibs);
				}

				exit(0);
			}
			else {
				delete(g_MemRegion_);

				if(g_nvlibs) {
					delete(g_nvlibs);
				}
			}
		};

		void run() {
			if(run_ == 1) return;
			run_ = 1;
		}

		int rank() { return rank_; }
	private:
		int run_;
		int rank_;
		int size_;
};

HUM_Runtime hum_runtime;
void hum_run() {
	hum_runtime.run();
}
