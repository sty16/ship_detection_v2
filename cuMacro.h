#ifndef CUMACRO_H_
#define CUMACRO_H_
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
#define BDIMX 32
#define BDIMY BDIMX
#define THREADSPACE 32768                                // 32KB each thread hava global memory as stack
#endif