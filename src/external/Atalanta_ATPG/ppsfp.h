#include <vector>
#include <utility>
// #include<string.h>
using namespace std;

typedef struct FAULT *FAULTPTR;
typedef struct GATE *GATEPTR;
typedef	int status;
typedef	int level;

typedef vector<char*> LIST;
typedef pair<int, int> FAULT_RECORD;
typedef vector<FAULT_RECORD> VEC;

void update_all(int npi);
void initGateStackAndFreach(int nog, int maxdpi, int npi);
void restoreOutput1FromOutput();
void evalGatesFromFreeStack();
level faultSimForStemGate(register GATEPTR gut, level observe, GATEPTR Dominator, int maxdpi);
level getFaultObserveFromInput(FAULTPTR pf, register GATEPTR gut);
level updateFaultObserveByFFRGate(register GATEPTR gut, FAULTPTR **pf, level stemobs);
int updateFaultObserveByPOGate(register GATEPTR gut, status *flag, int nbit, int tarray[]);
int updateFaultObserveByPOGate_NameList(register GATEPTR gut, status *flag, int nbit, int tarray[], VEC *detected_list);
int updateWholeStemObserveAndDetect(GATEPTR stem, status *flag, status flag2, int nbit, int tarray[]);
int updateWholeStemObserveAndDetect_NameList(GATEPTR stem, status *flag, status flag2, int nbit, int tarray[], VEC *detected_list);
int Fault1_Simulation(int nog, int maxdpi, int npi, int npo, int nstem, GATEPTR stem[], int nbit, int tarray[]);
int Fault1_Simulation_NameList(int nog, int maxdpi, int npi, int npo, int nstem, GATEPTR stem[], int nbit, int tarray[], VEC *detected_list);
void updateFaultyStack();
void update_free_gates(int npi);
void updateEvalStackByFaultyStack();
void updateFaultyAndDynamicStack(int npi);
int Fault0_Simulation(int nog, int maxdpi, int npi, int npo, int nstem, GATEPTR stem[], int nbit, int tarray[]);
int Fault0_Simulation_Namelist(int nog, int maxdpi, int npi, int npo, int nstem, GATEPTR stem[], int nbit, int tarray[], VEC *detected_list);
