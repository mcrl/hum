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

#include <sstream>
#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/DenseMap.h"

//#define __DEBUG__

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory CUDAPrepCategory("CUDA Preprocessor");
static llvm::cl::opt<std::string> OutputFileName("o", llvm::cl::cat(CUDAPrepCategory), llvm::cl::desc("output file name"));
static llvm::cl::opt<bool> ExportDeviceCodeOption("export-device", llvm::cl::cat(CUDAPrepCategory));

static std::string magicNumber = "0xffffffabcdef";

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
	public:
		MyASTVisitor(ASTContext& Context, Rewriter &CodeR,
				std::list<std::pair<FileID, SourceLocation>> *EmbedL) :
			Ctx(Context), CodeRewriter(CodeR), EmbedList(EmbedL) {
				ModifiedDeclList.clear();
				ModifiedExprList.clear();
			}

		// caculate offset to get the l_paren
		int getLParenOffset(SourceLocation SourceLoc) {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			const char* charData = SM.getCharacterData(SourceLoc);
			int offset = 1;

			while (charData[offset] != '(') {
				offset++;
			}
			return offset;
		}

		// caculate offset to get the less without space or newline
		int getLessOffset(SourceLocation SourceLoc) {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			const char* charData = SM.getCharacterData(SourceLoc);
			int offset = 1;
			int space = 0;

			while (charData[offset] != '<') {
				if (charData[offset] == ' ' || charData[offset] == '\n')
					space++;
				else
					space = 0;
				offset++;
			}
			return (offset - space);
		}

		// caculate offset to get the end of kernel parameter
		int getEndOfParmOffset(SourceLocation ParmLoc) {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			const char* charData = SM.getCharacterData(ParmLoc);
			int offset = 1;

			while (charData[offset] != ',' && charData[offset] != ')') {
				offset++;
			}
			return offset;
		}

		bool VisitFunctionDecl(FunctionDecl *F) {
			LangOptions langOpts;
			SourceManager& SM = CodeRewriter.getSourceMgr();
			bool IsInFile = SM.isInMainFile(F->getLocation());

			DeclarationName DeclName = F->getNameInfo().getName();
			std::string FuncName = DeclName.getAsString();
			SourceLocation FuncNameLoc = F->getNameInfo().getLoc();
			SourceRange SR;
			std::string rewrittenFD;

			if (F->hasAttr<CUDAGlobalAttr>()) {
				std::string fileLocString = SM.getFileLoc(F->getLocation()).printToString(SM);
				// do not touch function in this location
				if (fileLocString.find("thrust/") != std::string::npos) {
					return true;
				}

				// check if same declaration location is already modified
				std::vector<SourceLocation>::iterator I =
					std::find(ModifiedDeclList.begin(), ModifiedDeclList.end(), F->getLocation());
				if (I != ModifiedDeclList.end())
					return true;
				ModifiedDeclList.push_back(F->getLocation());

				// when function is template function, get whole text
				FunctionTemplateDecl *FTD = F->getDescribedFunctionTemplate();
				if (FTD)
					SR = SM.getExpansionRange(FTD->getSourceRange()).getAsRange();
				else
					SR = SM.getExpansionRange(F->getSourceRange()).getAsRange();

				// when function has no body, put ';' at the end of the function
				if (!F->hasBody()) {
					CodeRewriter.InsertTextAfter(F->getEndLoc().getLocWithOffset(1), ";");
				}

				// next, duplicate function declaration before further manipulation
				rewrittenFD = CodeRewriter.getRewrittenText(SR);
				CodeRewriter.InsertTextAfter(
						SR.getEnd().getLocWithOffset(1), "\n\n" + rewrittenFD);

				// change function name to *__hum__
				SourceLocation postFixLoc = FuncNameLoc.getLocWithOffset(
						F->getNameInfo().getAsString().length());
				CodeRewriter.InsertTextAfter(postFixLoc, "__hum__");

				SourceLocation ArgLoc = FuncNameLoc.getLocWithOffset(getLParenOffset(FuncNameLoc) + 1);
				// add magic number and __numArgs
				CodeRewriter.InsertTextAfter(ArgLoc, "size_t __magic_number, size_t __numArgs, \n");

				// add argument info
				CodeRewriter.InsertTextAfter(ArgLoc, "void* arg_infos\n");
				if (F->getNumParams() > 0) {
					CodeRewriter.InsertTextAfter(ArgLoc, ", ");
				} 

				// add to embed list
				if (!IsInFile) {
					FileID includeFile = SM.getFileID(F->getLocation());
					SourceLocation includeLoc = SM.getIncludeLoc(includeFile);

					std::pair<FileID, SourceLocation> eItem = std::make_pair(includeFile, includeLoc);
					std::list<std::pair<FileID, SourceLocation>>::iterator I =
						std::find(EmbedList->begin(), EmbedList->end(), eItem);

					if (I == EmbedList->end()) {
						EmbedList->push_back(eItem);
					}
				}
			}

			return true;
		}

		uint64_t getFieldOffset(FieldDecl *CurrentField) {
			RecordDecl* RD = CurrentField->getParent();

			uint64_t Offset = 0;
			for (RecordDecl::field_iterator FI = RD->field_begin(),
					FE = RD->field_end(); FI != FE; ++FI) {
				FieldDecl* FD = *FI;

				QualType FieldType = FD->getType();
				assert(!FieldType->isIncompleteType());

				uint64_t Align = Ctx.getTypeAlignInChars(FieldType).getQuantity();
				Offset = (Offset + Align - 1) / Align * Align;
				if (FD == CurrentField) {
					break;
				}

				uint64_t Size = Ctx.getTypeSizeInChars(FieldType).getQuantity();
				Offset += Size;
			}
			return Offset;
		}

		int InspectStructureType(QualType argType, uint64_t structOffset, std::string *outStr) {
			int numMembers = 0;

			if (argType->isStructureType()) {
				const RecordType *RT = argType->getAs<RecordType>();
				std::string membInfoText = "";

				for (RecordDecl::field_iterator FI = RT->getDecl()->field_begin(),
						FE = RT->getDecl()->field_end(); FI != FE; ++FI) {
					FieldDecl *FD = *FI;
					uint64_t FieldOffset = getFieldOffset(FD);
					QualType FieldType = FD->getType();

					if (FieldType->isStructureType()) {
						numMembers += InspectStructureType(FieldType, structOffset + FieldOffset, &membInfoText);
					} else {
						// offset indicator
						membInfoText += ", " + std::to_string(structOffset + FieldOffset);

						// type indicator
						if (FieldType->isPointerType()) {
							membInfoText += ", 1";
						}
						else {
							membInfoText += ", 0";
						}

						// size indicator
						membInfoText += ", " + std::to_string(Ctx.getTypeSize(FieldType) / 8);

						numMembers++;
					}
				}

				outStr->append(membInfoText);
			}

			return numMembers;
		}

		bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
			LangOptions langOpts;
			PrintingPolicy printPolicy(langOpts);
			SourceManager& SM = CodeRewriter.getSourceMgr();
			bool IsInFile = SM.isInMainFile(E->getExprLoc());
			std::string kernelName;
			SourceRange SR;
			int numArgInfos = 0;

			std::string fileLocString = SM.getFileLoc(E->getExprLoc()).printToString(SM);
			// do not touch call in this location
			if (fileLocString.find("thrust/") != std::string::npos) {
				return true;
			}

			// check if same expression location is already modified
			std::vector<SourceLocation>::iterator I =
				std::find(ModifiedExprList.begin(), ModifiedExprList.end(), E->getExprLoc());
			if (I != ModifiedExprList.end())
				return true;
			ModifiedExprList.push_back(E->getExprLoc());

			if (DeclRefExpr *Callee = dyn_cast<DeclRefExpr>(E->getCallee())) {
				if (Callee->getType()->isTemplateTypeParmType()) {
					return true;
				} 
			} else if (CXXDependentScopeMemberExpr *Callee = dyn_cast<CXXDependentScopeMemberExpr>(E->getCallee())) {
				if (Callee->getBase()->getType()->isTemplateTypeParmType()) {
					return true;
				}
			}


			// duplicate original expression before further manipulation
			SR = SM.getExpansionRange(E->getSourceRange()).getAsRange();
			std::string legacyExpr = "\n#else\n" + CodeRewriter.getRewrittenText(SR) + ";\n#endif\n}";
			CodeRewriter.InsertTextAfter(
					SR.getEnd().getLocWithOffset(2), legacyExpr);

			// get kernel name
			if (UnresolvedLookupExpr* ULE = dyn_cast<UnresolvedLookupExpr>(E->getCallee())) {
				kernelName = ULE->getName().getAsString();
			}
			else if (E->getDirectCallee()) {
				kernelName = E->getDirectCallee()->getNameAsString();
			} else {
				assert(false);
				return true;
			}

			// change function name to *__hum__
			SourceLocation postFixLoc = E->getExprLoc().getLocWithOffset(
					getLessOffset(E->getExprLoc()));
			CodeRewriter.InsertTextAfter(postFixLoc, "__hum__");

			// add number of arguments variable after kernel config
			SourceLocation ConfigLoc = E->getConfig()->getEndLoc();
			SourceLocation ArgLoc = ConfigLoc.getLocWithOffset(getLParenOffset(ConfigLoc) + 1);
			CodeRewriter.InsertTextAfter(ArgLoc, magicNumber + ", " + std::to_string(E->getNumArgs()) + ", ");

			// add additional arguments
			if (E->getNumArgs() > 0) {
				CodeRewriter.InsertTextAfter(ArgLoc,
						"&" + kernelName + "_arg_infos" + ", ");
			} else {
				CodeRewriter.InsertTextAfter(ArgLoc, "NULL");
			}

			// kernel arguments
			std::string argInfoText = "";
			for (unsigned int i = 0; i < E->getNumArgs(); ++i) {
				Expr* kernelArg = E->getArg(i);
				QualType argType = kernelArg->getType(); 

				std::string argName = Lexer::getSourceText(
						CharSourceRange::getTokenRange(kernelArg->getSourceRange()), SM, LangOptions(), 0);

				if (i != 0)
					argInfoText += ", ";

				// type indicator
				if (argType->isStructureType()) {
					argInfoText += "2";
				} else if (argType->isPointerType()) {
					argInfoText += "1";
				} else {
					argInfoText += "0";
				}

				// size indicator
				if (argType->isTemplateTypeParmType() ||
						argType->isDependentType()) {
					argInfoText += ", sizeof(" + argName + ")";
				}
				else {
					argInfoText += ", " + std::to_string(Ctx.getTypeSize(argType) / 8);
				}

				numArgInfos += 2;

				// addtional processing for structure type
				std::string structInfoText = "";
				int numStructMembs = InspectStructureType(argType, 0, &structInfoText);
				if (numStructMembs) {
					argInfoText += ", " + std::to_string(numStructMembs) + structInfoText;
					numArgInfos += 1 + numStructMembs * 3; 
				} else if (argType->isStructureType()) {
					argInfoText += ", " + std::to_string(0) + structInfoText;
					numArgInfos += 1;
				}
			}

			// add arg info array
			if (E->getNumArgs() > 0) {
				argInfoText = "int " + kernelName + "_arg_infos[" + std::to_string(numArgInfos)
					+ "] = {" + argInfoText + "};";
			}
			CodeRewriter.InsertTextBefore(SM.getExpansionLoc(E->getBeginLoc()),
					"{\n#ifdef HUM_EXT_KERNEL\n" + argInfoText + "\n");

			// add to embed list
			if (!IsInFile) {
				FileID includeFile = SM.getFileID(E->getExprLoc());
				SourceLocation includeLoc = SM.getIncludeLoc(includeFile);

				std::pair<FileID, SourceLocation> eItem = std::make_pair(includeFile, includeLoc);
				std::list<std::pair<FileID, SourceLocation>>::iterator I =
					std::find(EmbedList->begin(), EmbedList->end(), eItem);

				if (I == EmbedList->end()) {
					EmbedList->push_back(eItem);
				}
			}

#ifdef __DEBUG__
			std::string target = CodeRewriter.getRewrittenText(SM.getExpansionRange(E->getSourceRange()).getAsRange());
		    if (target.find("__hum__") == std::string::npos) {
				llvm::outs() << target << "\n";
			}
#endif

			return true;
		}

	private:
		ASTContext& Ctx;
		Rewriter &CodeRewriter;
		std::list<std::pair<FileID, SourceLocation>> *EmbedList;
		std::vector<SourceLocation> ModifiedDeclList;
		std::vector<SourceLocation> ModifiedExprList;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class MyASTConsumer : public ASTConsumer {
	public:
		MyASTConsumer(ASTContext& Context, Rewriter &CodeR,
				std::list<std::pair<FileID, SourceLocation>> *EmbedL) :
			Visitor(Context, CodeR, EmbedL) {}

		// Override the method that gets called for each parsed top-level
		// declaration.
		bool HandleTopLevelDecl(DeclGroupRef DR) override {
			for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
				Decl* TLD = *b;
				Visitor.TraverseDecl(TLD);
			}
			return true;
		}

	private:
		MyASTVisitor Visitor;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
	public:
		MyFrontendAction() {
			embedList.clear();
		}

		std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
				StringRef file) override {
			//llvm::errs() << "** Creating AST consumer for: " << file << "\n";
			CodeRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

			return std::make_unique<MyASTConsumer>(
					CI.getASTContext(), CodeRewriter, &embedList);
		}

		// caculate offset to get the begin of include macro
		int getBeginOfIncludeOffset(SourceLocation includeLoc) {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			int offset = -1;
			const char* charData = SM.getCharacterData(includeLoc.getLocWithOffset(offset));

			while (charData[0] != '#') {
				offset--;
				charData = SM.getCharacterData(includeLoc.getLocWithOffset(offset));
			}

			return offset;
		}

		// caculate offset to get the end of include macro
		int getEndOfIncludeOffset(SourceLocation includeLoc) {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			const char* charData = SM.getCharacterData(includeLoc);
			int offset = 1;

			// when file is included using ""
			if (charData[0] == '<') {
				while (charData[offset] != '>') {
					offset++;
				}
			}

			// when file is included using <>
			else if (charData[0] == '"') {
				while (charData[offset] != '"') {
					offset++;
				}
			}

			else {
				while (charData[offset] != '\n') {
					offset++;
				}
				//llvm::errs() << "Cannot calculate end offset of " << includeLoc.printToString(SM) << "\n";
				//exit(1);
			}

			return offset;
		}

		void CheckIncludeChain() {
			SourceManager& SM = CodeRewriter.getSourceMgr();

			// check if files in embed list is included by other file or main file
			for (std::list<std::pair<FileID, SourceLocation>>::iterator
					I = embedList.begin(), E = embedList.end(); I != E; ++I) {
				SourceLocation itemLoc = I->second;
				FileID itemFileID = SM.getFileID(itemLoc);

				if (itemFileID == SM.getMainFileID())
					continue;

				bool includeFound = false;
				for (std::list<std::pair<FileID, SourceLocation>>::iterator
						II = embedList.begin(), IE = embedList.end(); II != IE; ++II) {
					FileID checkFileID = II->first;
					if (itemFileID == checkFileID) {
						includeFound = true;
						break;
					}
				}

				if (!includeFound) {
					FileID includeFile = itemFileID;
					SourceLocation includeLoc = SM.getIncludeLoc(includeFile);
					std::pair<FileID, SourceLocation> eItem = std::make_pair(includeFile, includeLoc);
					embedList.push_back(eItem);

					// update the end of the embed list
					E = embedList.end();
				}
			}
		}

		void EmbedIncludedFiles() {
			SourceManager& SM = CodeRewriter.getSourceMgr();
			LangOptions langOpts;
			SourceRange SR;
			std::string rewrittenFile;

			CheckIncludeChain();

			//llvm::errs() << "EmbedList size=" << embedList.size() << "\n";

			for (std::list<std::pair<FileID, SourceLocation>>::iterator
					I = embedList.begin(), E = embedList.end(); I != E; ++I) {
				FileID fileID = I->first;
				SourceLocation includeLoc = I->second;

				//llvm::errs() << "Embedding code at " << includeLoc.printToString(SM) << "\n";

				SourceLocation includeBegin =
					includeLoc.getLocWithOffset(getBeginOfIncludeOffset(includeLoc));
				SourceLocation includeEnd =
					includeLoc.getLocWithOffset(getEndOfIncludeOffset(includeLoc));

				// delete include macro
				SourceRange includeRange(includeBegin, includeEnd);
				CodeRewriter.RemoveText(includeRange);

				// embed the include target code
				SR = SourceRange(SM.getLocForStartOfFile(fileID), SM.getLocForEndOfFile(fileID));
				rewrittenFile = CodeRewriter.getRewrittenText(SR);
				CodeRewriter.InsertTextAfter(includeBegin, "\n\n" + rewrittenFile);
			}
		}

		void EndSourceFileAction() override {
			EmbedIncludedFiles();

			std::error_code EC;
			SourceManager &CodeSM = CodeRewriter.getSourceMgr();

			SmallString<128> fullName(CodeSM.getFileEntryForID(CodeSM.getMainFileID())->getName());
			llvm::sys::path::replace_extension(fullName, "");

			std::string newFileName;

			if (OutputFileName == "") {
				newFileName = fullName.c_str();
				newFileName += ".tmp.cu";
			} else {
				newFileName = OutputFileName;
			}

			llvm::raw_fd_ostream outCode(newFileName.c_str(), EC, llvm::sys::fs::F_Text);

			if (EC) {
				llvm::errs() << EC.message() << "\n";
				exit(-1);
			}
			outCode.SetUnbuffered();

			// Now emit the rewritten buffer.
			//CodeRewriter.getEditBuffer(CodeSM.getMainFileID()).write(llvm::outs());
			CodeRewriter.getEditBuffer(CodeSM.getMainFileID()).write(outCode);
		}

	private:
		Rewriter CodeRewriter;
		std::list<std::pair<FileID, SourceLocation>> embedList;
};

int main(int argc, const char **argv) {
	CommonOptionsParser op(argc, argv, CUDAPrepCategory);
	ClangTool Tool(op.getCompilations(), op.getSourcePathList());

	// ClangTool::run accepts a FrontendActionFactory, which is then used to
	// create new objects implementing the FrontendAction interface. Here we use
	// the helper newFrontendActionFactory to create a default factory that will
	// return a new MyFrontendAction object every time.
	// To further customize this, we could create our own factory class.
	return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
