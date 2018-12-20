#include "stdafx.h"

#include "word2vec.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& OutFile,
	int& Dimensions, int& WinSize, int& Iter,
	bool& Verbose, TStr& InitInFile, TStr& DefaultEmbFile, bool& Sticky, bool& CustomDefault) 
{
	Env = TEnv(argc, argv, TNotify::StdNotify);
	Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
	InFile = Env.GetIfArgPrefixStr("-i:", "graph/test.walks",
	"Input walks path");
	InitInFile = Env.GetIfArgPrefixStr("-ie:", "", "Initial embeddings input file. Default is None.");
	DefaultEmbFile = Env.GetIfArgPrefixStr("-de:", "", "Default initial embeddings input file. Default is None.");
	OutFile = Env.GetIfArgPrefixStr("-o:", "emb/karate.emb",
	"Output embeddings path");
	Dimensions = Env.GetIfArgPrefixInt("-d:", 128,
	"Number of dimensions. Default is 128");
	WinSize = Env.GetIfArgPrefixInt("-k:", 10,
	"Context size for optimization. Default is 10");
	Iter = Env.GetIfArgPrefixInt("-e:", 1,
	"Number of epochs in SGD. Default is 1");
	Verbose = Env.IsArgStr("-v", "Verbose output.");
	Sticky = Env.IsArgStr("-s", "Using \"sticky\" factor.");
	if (InitInFile.Len() != 0)
		printf("Using node-specific initial embeddings.\n");
	if (DefaultEmbFile.Len() != 0)
	{
		printf("Using custom default embedding values\n\n");
		CustomDefault = true;
	}
	else
		CustomDefault = false;
}

//read walks from input file
//words/nodes should be represented by integers, excluding 0 (used as sentinel by word2vec code - I think)
void ReadWalks(TStr& InFile, bool& Verbose, TVVec<TInt, int64>& WalksVV) 
{
	TFIn FIn(InFile);
	int64 WalkCnt = 0;
	int64 NumWalks, MaxLen;

	try 
	{
		//read the first line: number of walks to read and max length
		TStr Line;
		TStrV Tokens;
		FIn.GetNextLn(Line);
		Line.SplitOnWs(Tokens);

		NumWalks = Tokens[0].GetInt();
		MaxLen = Tokens[1].GetInt();
		printf("Reading %lld walks, with max length %lld.\n", (long long)NumWalks, (long long)MaxLen);

		//size the walk container
		WalksVV = TVVec<TInt, int64>(NumWalks, MaxLen);

		//read remaining lines, one per walk
		while (!FIn.Eof()) 
		{
			FIn.GetNextLn(Line);
			Line.SplitOnWs(Tokens);

			//loop tokens (words/nodes)
			int word;
			for (int i = 0; i < Tokens.Len(); i++)
			{
				word = Tokens[i].GetInt();
				WalksVV.PutXY(WalkCnt, i, word);
			}

			WalkCnt++;
		}
		if (Verbose) { printf("Read %lld walks from %s successfully\n", (long long)WalkCnt, InFile.CStr()); }
	} 
	catch (PExcept Except) 
	{
		printf("bad fail\n");
		if (Verbose) 
		{
			printf("Read %lld walks from %s, then %s\n", (long long)WalkCnt, InFile.CStr(),
			Except->GetStr().CStr());
		}
	}
}

//read initial embeddings values from file, save to hash
//assumes the sticky factors given are quality, so flip them (1-val) to get true sticky factor
void ReadInitialEmbeddings(TStr& InitInFile, TStr& DefaultEmbFile, TIntFltVH& InitEmbeddingsHV, bool& Sticky, TIntFltH& StickyFactorsH, TFltV& DefaultEmbeddingV, TFltV& EmbeddingVariabilityV, bool& Verbose, int Dimensions)
{
	//read node-specific initial embeddings
	if (InitInFile.Len() != 0)
	{
		TFIn FIn(InitInFile);
		int64 LineCnt = 0;
		try 
		{
			while (!FIn.Eof()) 
			{
				//get next line of file
				TStr Ln;
				FIn.GetNextLn(Ln);

				//split out comments
				TStr Line, Comment;
				Ln.SplitOnCh(Line,'#',Comment);

				//tokenize the line
				TStrV Tokens;
				Line.SplitOnWs(Tokens);

				//too few tokens, skip this line
				if(Tokens.Len() < Dimensions+1){ continue; }

				//extract tokens
				int64 NId = Tokens[0].GetInt();		//node id
				//printf("%ld ", NId);

				//params for this node
				TFltV CurrV(Dimensions);
				for (int i = 0; i < Dimensions; i++)
				{
					//get embedding value
					CurrV[i] = Tokens[i+1].GetFlt();
					//printf("%f ", CurrV[i]);
					
				}
				InitEmbeddingsHV.AddDat(NId, CurrV);	//add vector to this node's initial embeddings

				//sticky factor if we have it
				if (Sticky && Tokens.Len() >= Dimensions+2)
				{
					TFlt CurrStick = 1 - Tokens[Dimensions+1].GetFlt();
					//printf("(%f)", CurrStick);
					StickyFactorsH.AddDat(NId, CurrStick);
				}

				//printf("\n");
				LineCnt++;
			}
			if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, InitInFile.CStr()); }
		} 
		catch (PExcept Except) 
		{
			if (Verbose) 
			{
				printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, InitInFile.CStr(),
				Except->GetStr().CStr());
			}
		}
	}

	//read default values for non-specified initial embeddings
	if (DefaultEmbFile.Len() != 0)
	{
		TFIn FIn(DefaultEmbFile);
		try 
		{
			//one line per embedding dimensions, each with two values
			//default embedding value for this position, and allowable variability percentage
			//example:		0.5 0.15		set embedding to 0.5, += 15% of 0.5
			for (int64 i = 0; i < Dimensions; i++)
			{
				//get next line of file
				TStr Line;
				FIn.GetNextLn(Line);

				//tokenize the line
				TStrV Tokens;
				Line.SplitOnWs(Tokens);

				//extract values
				DefaultEmbeddingV[i] = Tokens[0].GetFlt();			//embedding value
				EmbeddingVariabilityV[i] = Tokens[1].GetFlt();		//variability percentage
				//printf("%f %f\n", DefaultEmbeddingV[i], EmbeddingVariabilityV[i]);
			}
			if (Verbose) { printf("Read default embedding values\n"); }
		} 
		catch (PExcept Except) 
		{
			if (Verbose) 
			{
				printf("Reading from %s, then %s\n", InitInFile.CStr(),
				Except->GetStr().CStr());
			}
		}
	}
	
}

//dump embeddings (and optional walks) to output file
void WriteOutput(TStr& OutFile, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV,
 bool& OutputWalks) 
{
	printf("function\n");
	TFOut FOut(OutFile);
	if (OutputWalks) 
	{
		printf("outputting walks\n");
		for (int64 i = 0; i < WalksVV.GetXDim(); i++) 
		{
			for (int64 j = 0; j < WalksVV.GetYDim(); j++) 
			{
				FOut.PutInt(WalksVV(i,j));
				if(j+1==WalksVV.GetYDim()) 
				{
					FOut.PutLn();
				} 
				else 
				{
					FOut.PutCh(' ');
				}
			}
		}
		return;
	}
	/*
	bool First = 1;
	for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) 
	{
		if (First) 
		{
			FOut.PutInt(EmbeddingsHV.Len());
			FOut.PutCh(' ');
			FOut.PutInt(EmbeddingsHV[i].Len());
			FOut.PutLn();
			First = 0;
		}
		FOut.PutInt(EmbeddingsHV.GetKey(i));
		for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) 
		{
			FOut.PutCh(' ');
			FOut.PutFlt(EmbeddingsHV[i][j]);
		}
		FOut.PutLn();
	}
	*/
}

int main(int argc, char* argv[])
{
	TStr InFile, InitInFile, DefaultEmbFile, OutFile;
	int Dimensions, WinSize, Iter;
	bool Verbose, Sticky, CustomDefault;

	//parse command line args
	ParseArgs(argc, argv, InFile, OutFile, Dimensions, WinSize,
	Iter, Verbose, InitInFile, DefaultEmbFile, Sticky, CustomDefault);

	TIntFltVH EmbeddingsHV;			//embeddings object - hash int to vector of floats
	TVVec <TInt, int64> WalksVV;	//walks
	TIntFltVH InitEmbeddingsHV;		//initial embedding object setting: hash int to vector of floats
	TIntFltH StickyFactorsH;		//sticky factors: hash int node id to float
	TFltV DefaultEmbeddingV(Dimensions), EmbeddingVariabilityV(Dimensions);		//default embedding value and associated variability percentage (one pair per dimension)

	ReadWalks(InFile, Verbose, WalksVV);		//read walks from input file

	//read initial embeddings and/or default embedding values
	if (InitInFile.Len() != 0 or DefaultEmbFile.Len() != 0)
		ReadInitialEmbeddings(InitInFile, DefaultEmbFile, InitEmbeddingsHV, Sticky, StickyFactorsH, DefaultEmbeddingV, EmbeddingVariabilityV, Verbose, Dimensions);

	//run word2vec: network, configuration parameters, objects for walks and embeddings
	/*
	LearnEmbeddings(TVVec<TInt, int64>& WalksVV, const int& Dimensions,
  const int& WinSize, const int& Iter, const bool& Verbose,
  TIntFltVH& EmbeddingsHV, TIntFltVH& InitEmbeddingsHV, TIntFltH& StickyFactorsH, const bool& CustomDefault, TFltV& DefaultEmbeddingV, TFltV& EmbeddingVariabilityV)
  */


	//dump results
	bool temp = true;
	WriteOutput(OutFile, EmbeddingsHV, WalksVV, temp);
	printf("Results written to output file.\n");
	return 0;
}
