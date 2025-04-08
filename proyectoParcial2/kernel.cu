#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

struct Agent {
	float x, y;
	float P_con, P_ext, P_fat; // Probabilidades de contagio, contagio externo y mortalidad
	float P_mov, P_smo; // Probabilidades de movimiento y social distancing
	int T_inc, T_rec;
	int S; // Estado
};

const int N = 1024;
const int d_Max = 30; // Max de días
const int m_Max = 10; // Max de movimientos
const float R = 1.0; // Radio de contagio
const float l_Max = 5.0; // Radio de movimiento local
const float p = 500.0f, q = 500.0f; // Tamaño del espacio

void initializedAgents(Agent* agents) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis_p(0.0f, p);
	uniform_real_distribution<float> dis_q(0.0f, q);

	for (int i = 0; i < N; i++) {
		agents[i].x = dis_p(gen);
		agents[i].y = dis_q(gen);
		agents[i].P_con = 0.02f + static_cast <float> (rand()) / (RAND_MAX / (0.01f)); // [0.02, 0.03]
		agents[i].P_ext = 0.02f + static_cast <float> (rand()) / (RAND_MAX / (0.01f)); // [0.02, 0.03]
		agents[i].P_fat = static_cast <float> (rand()) / RAND_MAX * 0.063f + 0.007f; // [0.007, 0.07]
		agents[i].P_mov = 0.3f + static_cast <float> (rand()) / (RAND_MAX / (0.2f)); // [0.3, 0.5]
		agents[i].P_smo = 0.7f + static_cast <float> (rand()) / (RAND_MAX / (0.2f)); // [0.7, 0.9]
		agents[i].T_inc = 5 + rand() % 2; // [5, 6]
		agents[i].T_rec = 14;
		agents[i].S = 0; // Inicialmente no afectados
	}

}


__global__ void kernelInicializar(Agent* agentes, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curand_init(1234, idx, 0, &states[idx]);  // Semilla fija para reproducibilidad

    agentes[idx].x = curand_uniform(&states[idx]) * p;
    agentes[idx].y = curand_uniform(&states[idx]) * q;
    agentes[idx].P_con = 0.02f + curand_uniform(&states[idx]) * 0.01f;
    // ... Inicializar otros atributos similarmente
}


__global__ void regla1(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || agentes[i].S != 0) return;  // Solo agentes no infectados

    for (int j = 0; j < N; j++) {
        if (i != j && agentes[j].S > 0) {  // Vecino infectado
            float dx = agentes[i].x - agentes[j].x;
            float dy = agentes[i].y - agentes[j].y;
            float distancia = sqrtf(dx * dx + dy * dy);

            if (distancia <= R) {
                float rand_val = curand_uniform(&states[i]);
                if (rand_val <= agentes[i].P_con) {
                    agentes[i].S = 1;  // Infectar
                }
            }
        }
    }
}


void simularCPU(Agent* agentes) {
    for (int dia = 0; dia < d_Max; dia++) {
        for (int mov = 0; mov < m_Max; mov++) {
            // Aplicar Regla 1 y 2 para cada agente
        }
        // Aplicar Reglas 3-5
    }
}

int main() {
    // ------------ CPU ------------
    Agent* agentesCPU = new Agent[N];
    initializedAgents(agentesCPU);

    clock_t inicioCPU = clock();
    simularCPU(agentesCPU);
    clock_t finCPU = clock();

    // ------------ GPU ------------
    Agent* agentesGPU;
    curandState* estadosGPU;
    cudaMalloc(&agentesGPU, N * sizeof(Agent));
    cudaMalloc(&estadosGPU, N * sizeof(curandState)));

    // Configurar kernels
    dim3 bloques((N + 255) / 256);
    dim3 hilos(256);

    kernelInicializar << <bloques, hilos >> > (agentesGPU, estadosGPU);

    cudaEvent_t inicioGPU, finGPU;
    cudaEventCreate(&inicioGPU);
    cudaEventCreate(&finGPU);

    cudaEventRecord(inicioGPU);
    // Ejecutar kernels para cada regla en GPU
    regla1 << <bloques, hilos >> > (agentesGPU, estadosGPU);
    // ... Llamar otros kernels
    cudaEventRecord(finGPU);
    cudaEventSynchronize(finGPU);

    // ------------ Resultados ------------
    float tiempoCPU = (finCPU - inicioCPU) / CLOCKS_PER_SEC;
    float tiempoGPU;
    cudaEventElapsedTime(&tiempoGPU, inicioGPU, finGPU);

    std::cout << "Tiempo CPU: " << tiempoCPU << "s\n";
    std::cout << "Tiempo GPU: " << tiempoGPU / 1000 << "s\n";

    // Liberar memoria
    delete[] agentesCPU;
    cudaFree(agentesGPU);
    cudaFree(estadosGPU);

    return 0;
}
