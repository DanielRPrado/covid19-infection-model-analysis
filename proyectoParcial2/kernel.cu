#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>
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

// Definiciones mediante macros (son visibles tanto en CPU como en GPU)
#define N 1024         // Número de agentes
#define d_Max 30       // Número máximo de días simulados
#define m_Max 10       // Número máximo de movimientos por día
#define R 1.0f         // Radio de contagio (metros)
#define l_Max 5.0f     // Radio máximo de movimiento local (metros)
#define p 500.0f       // Ancho del área simulada (metros)
#define q 500.0f       // Alto del área simulada (metros)

struct Stats {
    int total_infectados = 0;
    int total_recuperados = 0;
    int total_fallecidos = 0;
    int nuevos_contagios[d_Max] = { 0 };
    int nuevos_recuperados[d_Max] = { 0 };
    int nuevos_fallecidos[d_Max] = { 0 };

    int primer_contagio = -1;
    int mitad_contagios = -1;
    int total_contagios = -1;

    int primer_recuperado = -1;
    int mitad_recuperados = -1;
    int total_recuperados_dia = -1;

    int primer_fallecido = -1;
    int mitad_fallecidos = -1;
	int total_fallecidos_dia = -1;
};

struct Agent {
    float x, y;
    float P_con, P_ext, P_fat; // Probabilidades de contagio, contagio externo y mortalidad
    float P_mov, P_smo; // Probabilidades de movimiento y social distancing
    int T_inc, T_rec;
    int S; // Estado
};


void imprimirResultados(const Stats& stats, double tiempoCPU, double tiempoGPU);
void actualizarEstadisticasCPU(Stats& stats, Agent* agents, int dia);
/////////////// Funciones para la versión CPU ///////////////

// Inicializa los agentes en CPU usando el STL random
void initializedAgentsCPU(Agent* agents) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis_p(0.0f, p);
    uniform_real_distribution<float> dis_q(0.0f, q);
    uniform_real_distribution<float> dis_prob(0.0f, 1.0f);

    for (int i = 0; i < N; i++) {
        agents[i].x = dis_p(gen);
        agents[i].y = dis_q(gen);

        agents[i].P_con = 0.02f + dis_prob(gen) * 0.01f;   // [0.02, 0.03]
        agents[i].P_ext = 0.02f + dis_prob(gen) * 0.01f;   // [0.02, 0.03]
        agents[i].P_fat = 0.007f + dis_prob(gen) * 0.063f; // [0.007, 0.07]
        agents[i].P_mov = 0.3f + dis_prob(gen) * 0.2f;     // [0.3, 0.5]
        agents[i].P_smo = 0.7f + dis_prob(gen) * 0.2f;     // [0.7, 0.9]
        agents[i].T_inc = 5 + (rand() % 2);                // 5 o 6 días
        agents[i].T_rec = 14;
        agents[i].S = 0; // Inicia no infectado
    }
}

// Regla 1: Contagio interno (CPU)
// Para cada agente no infectado, se revisa si existe algún vecino infectado dentro del radio R
void regla1CPU(Agent* agents, int i) {
    if (agents[i].S != 0) return;  // Solo se evalúa si no está infectado

    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        // Si el vecino está infectado (estado 1) 
        if (agents[j].S == 1) {
            float dx = agents[i].x - agents[j].x;
            float dy = agents[i].y - agents[j].y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist <= R) {
                // Se genera un valor aleatorio y se compara con la probabilidad de contagio
                float r = static_cast<float>(rand()) / RAND_MAX;
                if (r <= agents[i].P_con) {
                    agents[i].S = 1; // Se infecta
                    agents[i].T_inc = (rand() % 2) + 5; // Reinicia el tiempo de incubación
                    break;
                }
            }
        }
    }
}

// Regla 2: Movilidad (CPU)
// Se decide el movimiento según la probabilidad P_mov y se decide el tipo de movimiento por P_smo
void regla2CPU(Agent* agents, int i) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    if (r <= agents[i].P_mov) {
        float t = static_cast<float>(rand()) / RAND_MAX;
        if (t <= agents[i].P_smo) {
            // Movimiento local
            float dx = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f) * l_Max;
            float dy = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f) * l_Max;
            agents[i].x += dx;
            agents[i].y += dy;
        }
        else {
            // Movimiento global: posición aleatoria
            agents[i].x = static_cast<float>(rand()) / RAND_MAX * p;
            agents[i].y = static_cast<float>(rand()) / RAND_MAX * q;
        }
        // Asegurar que la posición se mantenga dentro de los límites
        if (agents[i].x < 0) agents[i].x = 0;
        if (agents[i].x > p) agents[i].x = p;
        if (agents[i].y < 0) agents[i].y = 0;
        if (agents[i].y > q) agents[i].y = q;
    }
}

// Regla 3: Contagio externo (CPU)
// Si el agente no está infectado, se puede infectar por factores externos
void regla3CPU(Agent* agents, int i) {
    if (agents[i].S == 0) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        if (r <= agents[i].P_ext) {
            agents[i].S = 1;
            agents[i].T_inc = (rand() % 2) + 5;
        }
    }
}

// Regla 4: Progreso de la infección (CPU)
// Si se agota el tiempo de incubación, el agente pasa a cuarentena (estado -1) y luego se recupera (estado 2)
void regla4CPU(Agent* agents, int i) {
    if (agents[i].S == 1) {
        if (agents[i].T_inc > 0)
            agents[i].T_inc--;
        if (agents[i].T_inc == 0)
            agents[i].S = -1;  // En cuarentena
    }
    else if (agents[i].S == -1) {
        if (agents[i].T_rec > 0)
            agents[i].T_rec--;
        if (agents[i].T_rec == 0)
            agents[i].S = 2;   // Recuperado
    }
}

// Regla 5: Fatalidad (CPU)
// Durante la cuarentena, se evalúa si el agente fallece según P_fat
void regla5CPU(Agent* agents, int i) {
    if (agents[i].S == -1) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        if (r <= agents[i].P_fat)
            agents[i].S = -2; // Fallecido
    }
}

// Función de simulación en CPU: ejecuta la simulación por días y movimientos
void simularCPU(Agent* agents, Stats& stats) {
    int acum_infectados = 0;
    int acum_fallecidos = 0;
    int acum_recuperados = 0;
    for (int dia = 0; dia < d_Max; dia++) {
        // Para cada movimiento del día
        for (int mov = 0; mov < m_Max; mov++) {
            for (int i = 0; i < N; i++) {
                regla1CPU(agents, i);
                regla2CPU(agents, i);
            }
        }
        // Al final del día se aplican las reglas 3, 4 y 5
        for (int i = 0; i < N; i++) {
            regla3CPU(agents, i);
            regla4CPU(agents, i);
            regla5CPU(agents, i);
        }

        actualizarEstadisticasCPU(stats, agents, dia);

        if(stats.nuevos_contagios[dia] > 0 && stats.primer_contagio == -1) {
            stats.primer_contagio = dia;
		}

        acum_infectados += stats.nuevos_contagios[dia];
		if(acum_infectados >= N/2 && stats.mitad_contagios == -1) {
			stats.mitad_contagios = dia;
		}
        if(acum_infectados >= N && stats.total_contagios == -1) {
			stats.total_contagios = dia;
		}

        if(stats.nuevos_recuperados[dia] > 0 && stats.primer_recuperado == -1) {
			stats.primer_recuperado = dia;
		}
        acum_recuperados += stats.nuevos_recuperados[dia];

        if(acum_recuperados >= N/2 && stats.mitad_recuperados == -1) {
			stats.mitad_recuperados = dia;
		}
		if(acum_recuperados >= N && stats.total_recuperados_dia == -1) {
			stats.total_recuperados_dia = dia;
		}

		if(stats.nuevos_fallecidos[dia] > 0 && stats.primer_fallecido == -1) {
			stats.primer_fallecido = dia;
		}
		acum_fallecidos += stats.nuevos_fallecidos[dia];

		if(acum_fallecidos >= N/2 && stats.mitad_fallecidos == -1) {
			stats.mitad_fallecidos = dia;
		}
		if(acum_fallecidos >= N && stats.total_fallecidos_dia == -1) {
			stats.total_fallecidos_dia = dia;
		}
    }
}


//////////////////  Kernel para la versión GPU ///////////////
// Kernel para inicializar agentes en GPU y configurar estados del generador (curand)
__global__ void kernelInicializar(Agent* agentes, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curand_init(1234, idx, 0, &states[idx]);

    agentes[idx].x = curand_uniform(&states[idx]) * p;
    agentes[idx].y = curand_uniform(&states[idx]) * q;
    agentes[idx].P_con = 0.02f + curand_uniform(&states[idx]) * 0.01f;
    agentes[idx].P_ext = 0.02f + curand_uniform(&states[idx]) * 0.01f;
    agentes[idx].P_fat = 0.007f + curand_uniform(&states[idx]) * 0.063f;
    agentes[idx].P_mov = 0.3f + curand_uniform(&states[idx]) * 0.2f;
    agentes[idx].P_smo = 0.7f + curand_uniform(&states[idx]) * 0.2f;
    agentes[idx].T_inc = (curand_uniform(&states[idx]) < 0.5f ? 5 : 6);
    agentes[idx].T_rec = 14;
    agentes[idx].S = 0;
}

__global__ void actualizarEstadisticasGPU(Agent* agentes, int* d_stats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atomicAdd(&d_stats[0], (agentes[i].S == 1));   // Infectados
    atomicAdd(&d_stats[1], (agentes[i].S == 2));   // Recuperados
    atomicAdd(&d_stats[2], (agentes[i].S == -2));  // Fallecidos
}
// Kernel Regla 1: Contagio interno (GPU)
// Cada hilo revisa un agente y lo infecta si existe algún vecino en rango
__global__ void regla1GPU(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || agentes[i].S != 0) return;

    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        if (agentes[j].S == 1) {
            float dx = agentes[i].x - agentes[j].x;
            float dy = agentes[i].y - agentes[j].y;
            float distancia = sqrtf(dx * dx + dy * dy);
            if (distancia <= R) {
                float rand_val = curand_uniform(&states[i]);
                if (rand_val <= agentes[i].P_con) {
                    agentes[i].S = 1;
                    agentes[i].T_inc = (curand_uniform(&states[i]) < 0.5f ? 5 : 6);
                    break;
                }
            }
        }
    }
}

// Kernel Regla 2: Movilidad (GPU)
__global__ void regla2GPU(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float move_decision = curand_uniform(&states[i]);
    if (move_decision <= agentes[i].P_mov) {
        float type_decision = curand_uniform(&states[i]);
        if (type_decision <= agentes[i].P_smo) {
            float dx = (curand_uniform(&states[i]) * 2.0f - 1.0f) * l_Max;
            float dy = (curand_uniform(&states[i]) * 2.0f - 1.0f) * l_Max;
            agentes[i].x += dx;
            agentes[i].y += dy;
            // Aseguramos límites
            agentes[i].x = fminf(fmaxf(agentes[i].x, 0.0f), p);
            agentes[i].y = fminf(fmaxf(agentes[i].y, 0.0f), q);
        }
        else {
            agentes[i].x = curand_uniform(&states[i]) * p;
            agentes[i].y = curand_uniform(&states[i]) * q;
        }
    }
}

// Kernel Regla 3: Contagio externo (GPU)
__global__ void regla3GPU(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (agentes[i].S == 0) {
        float rand_ext = curand_uniform(&states[i]);
        if (rand_ext <= agentes[i].P_ext) {
            agentes[i].S = 1;
            agentes[i].T_inc = (curand_uniform(&states[i]) < 0.5f ? 5 : 6);
        }
    }
}

// Kernel Regla 4: Progreso de la infección (GPU)
__global__ void regla4GPU(Agent* agentes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (agentes[i].S == 1) {
        if (agentes[i].T_inc > 0)
            agentes[i].T_inc--;
        if (agentes[i].T_inc == 0)
            agentes[i].S = -1;
    }
    else if (agentes[i].S == -1) {
        if (agentes[i].T_rec > 0)
            agentes[i].T_rec--;
        if (agentes[i].T_rec == 0)
            agentes[i].S = 2;
    }
}

// Kernel Regla 5: Fatalidad (GPU)
__global__ void regla5GPU(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (agentes[i].S == -1) {
        float rand_val = curand_uniform(&states[i]);
        if (rand_val <= agentes[i].P_fat)
            agentes[i].S = -2;
    }
}

void simularGPU(Agent* d_agents, curandState* d_states, Stats& stats) {
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int* d_daily_stats;
    cudaMalloc(&d_daily_stats, 3 * sizeof(int));

    int acum_infectados = 0;
    int acum_recuperados = 0;
    int acum_fallecidos = 0;

    for (int dia = 0; dia < d_Max; dia++) {
        cudaMemset(d_daily_stats, 0, 3 * sizeof(int));

        for (int mov = 0; mov < m_Max; mov++) {
            regla1GPU << <blocks, threadsPerBlock >> > (d_agents, d_states);
            regla2GPU << <blocks, threadsPerBlock >> > (d_agents, d_states);
            cudaDeviceSynchronize();
        }

        regla3GPU << <blocks, threadsPerBlock >> > (d_agents, d_states);
        regla4GPU << <blocks, threadsPerBlock >> > (d_agents);
        regla5GPU << <blocks, threadsPerBlock >> > (d_agents, d_states);

        actualizarEstadisticasGPU << <blocks, threadsPerBlock >> > (d_agents, d_daily_stats);

        int daily_stats[3];
        cudaMemcpy(daily_stats, d_daily_stats, 3 * sizeof(int), cudaMemcpyDeviceToHost);

        stats.nuevos_contagios[dia] = daily_stats[0];
        stats.nuevos_recuperados[dia] = daily_stats[1];
        stats.nuevos_fallecidos[dia] = daily_stats[2];

        // Actualizar acumulados
        acum_infectados += daily_stats[0];
        acum_recuperados += daily_stats[1];
        acum_fallecidos += daily_stats[2];

        // Actualizar días clave
        if (daily_stats[0] > 0 && stats.primer_contagio == -1)
            stats.primer_contagio = dia;
        if (acum_infectados >= N / 2 && stats.mitad_contagios == -1)
            stats.mitad_contagios = dia;
        if (acum_infectados >= N && stats.total_contagios == -1)
            stats.total_contagios = dia;
    }
    cudaFree(d_daily_stats);
}

int main() {
    // Suimulación en CPU
    Agent* agentesCPU = new Agent[N];
    Stats statsCPU;
    initializedAgentsCPU(agentesCPU);

    auto startCPU = chrono::high_resolution_clock::now();
    simularCPU(agentesCPU, statsCPU);
    auto endCPU = chrono::high_resolution_clock::now();
    double tiempoCPU = chrono::duration<double>(endCPU - startCPU).count();

    // Simulación en GPU
    Agent* agentesGPU;
    curandState* estadosGPU;
    cudaMalloc(&agentesGPU, N * sizeof(Agent));
    cudaMalloc(&estadosGPU, N * sizeof(curandState));

    // Inicializar agentes en GPU
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelInicializar << <blocks, threadsPerBlock >> > (agentesGPU, estadosGPU);
    cudaDeviceSynchronize();

    Stats statsGPU;
    // Medir tiempos con eventos CUDA
    cudaEvent_t inicioGPU, finGPU;
    cudaEventCreate(&inicioGPU);
    cudaEventCreate(&finGPU);
    cudaEventRecord(inicioGPU);

	simularGPU(agentesGPU, estadosGPU, statsGPU);

    cudaEventRecord(finGPU);
    cudaEventSynchronize(finGPU);
    float tiempoGPU_ms;
    cudaEventElapsedTime(&tiempoGPU_ms, inicioGPU, finGPU);
    double tiempoGPU = tiempoGPU_ms / 1000.0;

    imprimirResultados(statsCPU, tiempoCPU, tiempoGPU);
    // Liberar memoria
    delete[] agentesCPU;
    cudaFree(agentesGPU);
    cudaFree(estadosGPU);

    return 0;
}

/////////////// Funciones auxiliares ///////////////
void actualizarEstadisticasCPU(Stats& stats, Agent* agents, int dia) {
    int contagios = 0;
    int recuperados = 0;
    int fallecidos = 0;

    for (int i = 0; i < N; i++) {
        contagios += (agents[i].S == 1);
        recuperados += (agents[i].S == 2);
        fallecidos += (agents[i].S == -2);
    }

    stats.nuevos_contagios[dia] = contagios;
    stats.nuevos_recuperados[dia] = recuperados;
    stats.nuevos_fallecidos[dia] = fallecidos;

    stats.total_infectados += contagios;
    stats.total_recuperados += recuperados;
    stats.total_fallecidos += fallecidos;
}

void imprimirResultados(const Stats& stats, double tiempoCPU, double tiempoGPU) {
    printf("\n=== RESULTADOS ===\n");
    printf("Tiempos de ejecucion:\n");
    printf("- CPU: %.2f segundos\n", tiempoCPU);
    printf("- GPU: %.2f segundos\n", tiempoGPU);
    printf("- Aceleracion GPU: %.1fx\n\n", tiempoCPU / tiempoGPU);

    printf("Casos acumulados:\n");
    printf("- Total infectados: %d\n", stats.total_infectados);
    printf("- Total recuperados: %d\n", stats.total_recuperados);
    printf("- Total fallecidos: %d\n\n", stats.total_fallecidos);

    printf("Dias clave (contagios):\n");
    printf("- Primer contagio: Dia %d\n", stats.primer_contagio + 1);
    printf("- 50%% de contagios: Dia %d\n", stats.mitad_contagios + 1);
    printf("- 100%% de contagios: Dia %d\n\n", stats.total_contagios + 1);

    printf("Dias clave (recuperados):\n");
    printf("- Primer recuperado: Dia %d\n", stats.primer_recuperado + 1);
    printf("- 50%% de recuperados: Dia %d\n", stats.mitad_recuperados + 1);
    printf("- 100%% de recuperados: Dia %d\n\n", stats.total_recuperados_dia + 1);

    printf("Dias clave (fallecidos):\n");
    printf("- Primer fallecido: Dia %d\n", stats.primer_fallecido + 1);
    printf("- 50%% de fallecidos: Dia %d\n", stats.mitad_fallecidos + 1);
    printf("- 100%% de fallecidos: Dia %d\n\n", stats.total_fallecidos_dia + 1);

    printf("Progresion diaria (5 primeros dias):\n");
    printf("Dia | Contagios | Recuperados | Fallecidos\n");
    for (int i = 0; i < 5; i++) {
        printf("%2d  | %6d    | %6d      | %6d\n",
            i + 1,
            stats.nuevos_contagios[i],
            stats.nuevos_recuperados[i],
            stats.nuevos_fallecidos[i]);
    }
}
