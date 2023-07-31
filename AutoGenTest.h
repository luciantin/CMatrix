#include "helpers/nnc_matrix.h"

NNCIMatrixType GetAutoGenTestMatrix(){
NNCIMatrixType zeroOneTest = NNCMatrixAlloc(2, 30);
zeroOneTest->matrix[0][0] = 0.0;zeroOneTest->matrix[0][1] = 0.0;
zeroOneTest->matrix[1][0] = 0.10738789290189743;zeroOneTest->matrix[1][1] = 0.02852226048707962;
zeroOneTest->matrix[2][0] = 0.09263824671506882;zeroOneTest->matrix[2][1] = -0.20199225842952728;
zeroOneTest->matrix[3][0] = -0.3222488760948181;zeroOneTest->matrix[3][1] = -0.08524538576602936;
zeroOneTest->matrix[4][0] = -0.3495118021965027;zeroOneTest->matrix[4][1] = 0.2745402753353119;
zeroOneTest->matrix[5][0] = -0.5210058689117432;zeroOneTest->matrix[5][1] = 0.19285966455936432;
zeroOneTest->matrix[6][0] = 0.5045865178108215;zeroOneTest->matrix[6][1] = 0.43570277094841003;
zeroOneTest->matrix[7][0] = 0.76882404088974;zeroOneTest->matrix[7][1] = 0.11767714470624924;
zeroOneTest->matrix[8][0] = 0.4926939308643341;zeroOneTest->matrix[8][1] = -0.7398487329483032;
zeroOneTest->matrix[9][0] = -0.7036499381065369;zeroOneTest->matrix[9][1] = -0.7105468511581421;
zeroOneTest->matrix[10][0] = -0.0;zeroOneTest->matrix[10][1] = -0.0;
zeroOneTest->matrix[11][0] = -0.07394106686115265;zeroOneTest->matrix[11][1] = 0.0829361081123352;
zeroOneTest->matrix[12][0] = 0.008080541156232357;zeroOneTest->matrix[12][1] = 0.22207525372505188;
zeroOneTest->matrix[13][0] = 0.2454816699028015;zeroOneTest->matrix[13][1] = 0.22549913823604584;
zeroOneTest->matrix[14][0] = 0.3836473822593689;zeroOneTest->matrix[14][1] = -0.22437813878059387;
zeroOneTest->matrix[15][0] = -0.008016093634068966;zeroOneTest->matrix[15][1] = -0.5554977059364319;
zeroOneTest->matrix[16][0] = -0.6606056690216064;zeroOneTest->matrix[16][1] = 0.08969160914421082;
zeroOneTest->matrix[17][0] = -0.7174547910690308;zeroOneTest->matrix[17][1] = 0.30032801628112793;
zeroOneTest->matrix[18][0] = 0.1729927510023117;zeroOneTest->matrix[18][1] = 0.8718927502632141;
zeroOneTest->matrix[19][0] = 0.6619341373443604;zeroOneTest->matrix[19][1] = 0.7495619654655457;
zeroOneTest->matrix[20][0] = -0.0;zeroOneTest->matrix[20][1] = 0.0;
zeroOneTest->matrix[21][0] = 0.05838184431195259;zeroOneTest->matrix[21][1] = -0.09453697502613068;
zeroOneTest->matrix[22][0] = -0.1368253380060196;zeroOneTest->matrix[22][1] = -0.17510437965393066;
zeroOneTest->matrix[23][0] = -0.2751694321632385;zeroOneTest->matrix[23][1] = -0.1881299912929535;
zeroOneTest->matrix[24][0] = 0.19194842875003815;zeroOneTest->matrix[24][1] = 0.40085741877555847;
zeroOneTest->matrix[25][0] = -0.16649487614631653;zeroOneTest->matrix[25][1] = 0.5300202369689941;
zeroOneTest->matrix[26][0] = 0.6666014194488525;zeroOneTest->matrix[26][1] = 0.009327448904514313;
zeroOneTest->matrix[27][0] = 0.4328209161758423;zeroOneTest->matrix[27][1] = -0.6462231278419495;
zeroOneTest->matrix[28][0] = -0.8729175329208374;zeroOneTest->matrix[28][1] = -0.16774514317512512;
zeroOneTest->matrix[29][0] = -0.6297622919082642;zeroOneTest->matrix[29][1] = 0.7767879366874695;
return zeroOneTest;
}

NNCIMatrixType GetAutoGenTruthMatrix(){
NNCIMatrixType zeroOneTest = NNCMatrixAlloc(1, 30);
zeroOneTest->matrix[0][0] = 0;
zeroOneTest->matrix[1][0] = 0;
zeroOneTest->matrix[2][0] = 0;
zeroOneTest->matrix[3][0] = 0;
zeroOneTest->matrix[4][0] = 0;
zeroOneTest->matrix[5][0] = 0;
zeroOneTest->matrix[6][0] = 0;
zeroOneTest->matrix[7][0] = 0;
zeroOneTest->matrix[8][0] = 0;
zeroOneTest->matrix[9][0] = 0;
zeroOneTest->matrix[10][0] = 1;
zeroOneTest->matrix[11][0] = 1;
zeroOneTest->matrix[12][0] = 1;
zeroOneTest->matrix[13][0] = 1;
zeroOneTest->matrix[14][0] = 1;
zeroOneTest->matrix[15][0] = 1;
zeroOneTest->matrix[16][0] = 1;
zeroOneTest->matrix[17][0] = 1;
zeroOneTest->matrix[18][0] = 1;
zeroOneTest->matrix[19][0] = 1;
zeroOneTest->matrix[20][0] = 2;
zeroOneTest->matrix[21][0] = 2;
zeroOneTest->matrix[22][0] = 2;
zeroOneTest->matrix[23][0] = 2;
zeroOneTest->matrix[24][0] = 2;
zeroOneTest->matrix[25][0] = 2;
zeroOneTest->matrix[26][0] = 2;
zeroOneTest->matrix[27][0] = 2;
zeroOneTest->matrix[28][0] = 2;
zeroOneTest->matrix[29][0] = 2;
return zeroOneTest;
}
