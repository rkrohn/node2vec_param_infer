MAIN = node2vec
DEPH = $(DRVSNAPADV)/n2v.h $(DRVSNAPADV)/word2vec.h $(DRVSNAPADV)/biasedrandomwalk.h
DEPCPP = $(DRVSNAPADV)/n2v.cpp $(DRVSNAPADV)/word2vec.cpp $(DRVSNAPADV)/biasedrandomwalk.cpp
CXXFLAGS += $(CXXOPENMP)

MAIN = word2vec
DEPH = $(DRVSNAPADV)/word2vec.h
DEPCPP = $(DRVSNAPADV)/word2vec.cpp
CXXFLAGS += $(CXXOPENMP)