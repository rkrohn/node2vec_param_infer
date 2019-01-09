MAINNODE = node2vec
DEPHNODE = $(DRVSNAPADV)/n2v.h $(DRVSNAPADV)/word2vec.h $(DRVSNAPADV)/biasedrandomwalk.h
DEPCPPNODE = $(DRVSNAPADV)/n2v.cpp $(DRVSNAPADV)/word2vec.cpp $(DRVSNAPADV)/biasedrandomwalk.cpp

MAINWORD = word2vec
DEPHWORD = $(DRVSNAPADV)/word2vec.h
DEPCPPWORD = $(DRVSNAPADV)/word2vec.cpp

CXXFLAGS += $(CXXOPENMP)
