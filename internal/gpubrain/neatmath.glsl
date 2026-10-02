// The activation functions of neat.Network, shared by every kernel that runs
// NEAT networks. `precise` keeps the compiler from fusing the multiply-adds,
// so the results stay those of the CPU but for the division in fast_tanh.
#ifndef NEATMATH_GLSL
#define NEATMATH_GLSL

#define MEMORY 4u

float fast_tanh(float x) {
	if (x > 3.0) return 1.0;
	if (x < -3.0) return -1.0;
	precise float x2 = x * x;
	precise float num = x * (27.0 + x2);
	precise float den = 27.0 + 9.0 * x2;
	return num / den;
}
float sig(float x) { precise float y = 0.5 + 0.5 * fast_tanh(2.45 * x); return y; }
float sig01(float x) { precise float y = 0.5 + 0.5 * fast_tanh(0.5 * x); return y; }

#endif
