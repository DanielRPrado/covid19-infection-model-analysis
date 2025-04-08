#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <sstream>

template <typename T>
std::string to_string_custom(T value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

using namespace std;

// -----------------------
// Data Structures & Consts
// -----------------------
struct Agent {
    float x, y;
    float P_con, P_ext, P_fat; // contagion, external contagion, and fatality probabilities
    float P_mov, P_smo;        // movement and short-range movement probabilities
    int T_inc, T_rec;
    int S;                   // State: 0 = not infected, 1 = infected (incubating), -1 = quarantined, 2 = recovered, -2 = deceased
};

#define N 1024
#define d_Max 30       // Maximum simulation days
#define m_Max 10       // Movements per day
#define R 1.0f         // Contagion radius (in meters)
#define l_Max 5.0f     // Maximum local movement radius (m)
#define p 500.0f       // Simulation area dimension (width)
#define q 500.0f       // Simulation area dimension (height)


// -----------------------
// CPU Initialization
// -----------------------
void initializedAgents(Agent* agents) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis_p(0.0f, p);
    uniform_real_distribution<float> dis_q(0.0f, q);

    for (int i = 0; i < N; i++) {
        agents[i].x = dis_p(gen);
        agents[i].y = dis_q(gen);
        agents[i].P_con = 0.02f + static_cast<float>(rand()) / (RAND_MAX / (0.01f)); // [0.02, 0.03]
        agents[i].P_ext = 0.02f + static_cast<float>(rand()) / (RAND_MAX / (0.01f)); // [0.02, 0.03]
        agents[i].P_fat = static_cast<float>(rand()) / RAND_MAX * 0.063f + 0.007f;    // [0.007, 0.07]
        agents[i].P_mov = 0.3f + static_cast<float>(rand()) / (RAND_MAX / (0.2f));      // [0.3, 0.5]
        agents[i].P_smo = 0.7f + static_cast<float>(rand()) / (RAND_MAX / (0.2f));      // [0.7, 0.9]
        agents[i].T_inc = 5 + rand() % 2;  // Incubation period of 5 or 6 days
        agents[i].T_rec = 14;              // Recovery period
        agents[i].S = 0;                   // Initially, not infected
    }
}

// -----------------------
// GPU Kernels
// -----------------------
__global__ void kernelInicializar(Agent* agentes, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Fixed seed for reproducibility.
    curand_init(1234, idx, 0, &states[idx]);

    // Initialize positions and one probability as an example.
    agentes[idx].x = curand_uniform(&states[idx]) * p;
    agentes[idx].y = curand_uniform(&states[idx]) * q;
    agentes[idx].P_con = 0.02f + curand_uniform(&states[idx]) * 0.01f;
    // You can initialize other parameters similarly if needed.
}

__global__ void regla1(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || agentes[i].S != 0) return;  // Only non-infected agents

    for (int j = 0; j < N; j++) {
        if (i != j && agentes[j].S > 0) {  // if neighbor j is infected (state > 0)
            float dx = agentes[i].x - agentes[j].x;
            float dy = agentes[i].y - agentes[j].y;
            float distancia = sqrtf(dx * dx + dy * dy);
            if (distancia <= R) {
                float rand_val = curand_uniform(&states[i]);
                if (rand_val <= agentes[i].P_con) {
                    agentes[i].S = 1;  // Infect the agent
                    break;
                }
            }
        }
    }
}

__global__ void regla2(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float move_decision = curand_uniform(&states[i]);
    if (move_decision <= agentes[i].P_mov) {
        float type_decision = curand_uniform(&states[i]);
        if (type_decision <= agentes[i].P_smo) {
            // Local movement: small displacement
            float dx = (curand_uniform(&states[i]) * 2.0f - 1.0f) * l_Max;
            float dy = (curand_uniform(&states[i]) * 2.0f - 1.0f) * l_Max;
            agentes[i].x += dx;
            agentes[i].y += dy;
            // Clamp positions within the simulation area.
            agentes[i].x = fminf(fmaxf(agentes[i].x, 0.0f), p);
            agentes[i].y = fminf(fmaxf(agentes[i].y, 0.0f), q);
        }
        else {
            // Far movement: jump to a random location
            agentes[i].x = curand_uniform(&states[i]) * p;
            agentes[i].y = curand_uniform(&states[i]) * q;
        }
    }
}
__global__ void regla3(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 

    if (agentes[i].S == 0) {
        float rand_ext = curand_uniform(&states[i]);
        if (rand_ext <= agentes[i].P_ext) {
            agentes[i].S = 1;
        }
    }
}

__global__ void regla4(Agent* agentes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;  

    if (agentes[i].S == 1) {
        if (agentes[i].T_inc > 0) {
            agentes[i].T_inc--; 
            if (agentes[i].T_inc == 0) {
                agentes[i].S = -1;
            }
        }
    }
    else if (agentes[i].S == -1) {
        if (agentes[i].T_rec > 0) {
            agentes[i].T_rec--;  
            if (agentes[i].T_rec == 0) {
                agentes[i].S = 2; 
            }
        }
    }
}

__global__ void regla5(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 

    if (agentes[i].S == -1) {
        float rand_val = curand_uniform(&states[i]);
        if (rand_val <= agentes[i].P_fat) {
            agentes[i].S = -2;
        }
    }
}


__global__ void regla3(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (agentes[i].S == 0) {
        float rand_ext = curand_uniform(&states[i]);
        if (rand_ext <= agentes[i].P_ext) {
            agentes[i].S = 1;  // External infection
        }
    }
}

__global__ void regla4(Agent* agentes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Update incubation and recovery
    if (agentes[i].S == 1) {
        if (agentes[i].T_inc > 0) {
            agentes[i].T_inc--;
            if (agentes[i].T_inc == 0)
                agentes[i].S = -1;  // Start quarantine
        }
    }
    else if (agentes[i].S == -1) {
        if (agentes[i].T_rec > 0) {
            agentes[i].T_rec--;
            if (agentes[i].T_rec == 0)
                agentes[i].S = 2;   // Recovered
        }
    }
}

__global__ void regla5(Agent* agentes, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // For agents in quarantine, evaluate fatality.
    if (agentes[i].S == -1) {
        float rand_val = curand_uniform(&states[i]);
        if (rand_val <= agentes[i].P_fat) {
            agentes[i].S = -2; // Agent dies.
        }
    }
}

// -----------------------
// CPU Simulation Function
// -----------------------
// Note: We compute daily new cases as the change in cumulative counts.
// An agent is considered "infected" if its state is not zero.
void simularCPU(Agent* agentes, vector<int>& dailyInfected,
    vector<int>& dailyRecovered, vector<int>& dailyFatal)
{
    // Setup a random generator.
    default_random_engine generator(time(nullptr));
    uniform_real_distribution<float> distribution(0.0, 1.0);

    int prevCumInf = 0, prevCumRec = 0, prevCumFat = 0;

    for (int day = 0; day < d_Max; day++) {
        // For each movement during the day:
        for (int mov = 0; mov < m_Max; mov++) {
            // --- Regla 1: contagion by contact ---
            for (int i = 0; i < N; i++) {
                if (agentes[i].S == 0) {
                    for (int j = 0; j < N; j++) {
                        if (i != j && agentes[j].S > 0) { // neighbor infected
                            float dx = agentes[i].x - agentes[j].x;
                            float dy = agentes[i].y - agentes[j].y;
                            float distancia = sqrt(dx * dx + dy * dy);
                            if (distancia <= R) {
                                float rand_val = distribution(generator);
                                if (rand_val <= agentes[i].P_con) {
                                    agentes[i].S = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // --- Regla 2: movement ---
            for (int i = 0; i < N; i++) {
                float move_decision = distribution(generator);
                if (move_decision <= agentes[i].P_mov) {
                    float type_decision = distribution(generator);
                    if (type_decision <= agentes[i].P_smo) {
                        float dx = (distribution(generator) * 2.0f - 1.0f) * l_Max;
                        float dy = (distribution(generator) * 2.0f - 1.0f) * l_Max;
                        agentes[i].x += dx;
                        agentes[i].y += dy;
                        if (agentes[i].x < 0) agentes[i].x = 0;
                        if (agentes[i].x > p) agentes[i].x = p;
                        if (agentes[i].y < 0) agentes[i].y = 0;
                        if (agentes[i].y > q) agentes[i].y = q;
                    }
                    else {
                        agentes[i].x = distribution(generator) * p;
                        agentes[i].y = distribution(generator) * q;
                    }
                }
            }
        }
        // --- Regla 3: External contagion ---
        for (int i = 0; i < N; i++) {
            if (agentes[i].S == 0) {
                float rand_ext = distribution(generator);
                if (rand_ext <= agentes[i].P_ext) {
                    agentes[i].S = 1;
                }
            }
        }
        // --- Regla 4: Incubation and Recovery ---
        for (int i = 0; i < N; i++) {
            if (agentes[i].S == 1) {
                if (agentes[i].T_inc > 0) {
                    agentes[i].T_inc--;
                    if (agentes[i].T_inc == 0)
                        agentes[i].S = -1;
                }
            }
            else if (agentes[i].S == -1) {
                if (agentes[i].T_rec > 0) {
                    agentes[i].T_rec--;
                    if (agentes[i].T_rec == 0)
                        agentes[i].S = 2;
                }
            }
        }
        // --- Regla 5: Fatalities ---
        for (int i = 0; i < N; i++) {
            if (agentes[i].S == -1) {
                float rand_val = distribution(generator);
                if (rand_val <= agentes[i].P_fat) {
                    agentes[i].S = -2;
                }
            }
        }

        // Calculate cumulative counts at end of day
        int cumInf = 0, cumRec = 0, cumFat = 0;
        for (int i = 0; i < N; i++) {
            if (agentes[i].S != 0) cumInf++;
            if (agentes[i].S == 2) cumRec++;
            if (agentes[i].S == -2) cumFat++;
        }
        dailyInfected.push_back(cumInf - prevCumInf);
        dailyRecovered.push_back(cumRec - prevCumRec);
        dailyFatal.push_back(cumFat - prevCumFat);

        prevCumInf = cumInf;
        prevCumRec = cumRec;
        prevCumFat = cumFat;
    }
}

// -----------------------
// GPU Simulation Function
// -----------------------
// For each simulation day, we launch the kernels (for movement then daily update).
void simularGPU(Agent* agentesGPU, curandState* estadosGPU,
    vector<int>& dailyInfected, vector<int>& dailyRecovered, vector<int>& dailyFatal)
{
    Agent* hostAgents = new Agent[N];
    int prevCumInf = 0, prevCumRec = 0, prevCumFat = 0;
    dim3 bloques((N + 255) / 256);
    dim3 hilos(256);

    for (int day = 0; day < d_Max; day++) {
        for (int mov = 0; mov < m_Max; mov++) {
            regla1 << <bloques, hilos >> > (agentesGPU, estadosGPU);
            regla2 << <bloques, hilos >> > (agentesGPU, estadosGPU);
        }
        regla3 << <bloques, hilos >> > (agentesGPU, estadosGPU);
        regla4 << <bloques, hilos >> > (agentesGPU);
        regla5 << <bloques, hilos >> > (agentesGPU, estadosGPU);
        cudaDeviceSynchronize();

        // Copy device agents to host to compute daily statistics.
        cudaMemcpy(hostAgents, agentesGPU, N * sizeof(Agent), cudaMemcpyDeviceToHost);

        int cumInf = 0, cumRec = 0, cumFat = 0;
        for (int i = 0; i < N; i++) {
            if (hostAgents[i].S != 0) cumInf++;
            if (hostAgents[i].S == 2) cumRec++;
            if (hostAgents[i].S == -2) cumFat++;
        }
        dailyInfected.push_back(cumInf - prevCumInf);
        dailyRecovered.push_back(cumRec - prevCumRec);
        dailyFatal.push_back(cumFat - prevCumFat);

        prevCumInf = cumInf;
        prevCumRec = cumRec;
        prevCumFat = cumFat;
    }
    delete[] hostAgents;
}

// -----------------------
// Statistics Reporting Function
// -----------------------
void printSimulationStats(const vector<int>& dailyInfected,
    const vector<int>& dailyRecovered,
    const vector<int>& dailyFatal,
    double cpuTime, double gpuTime)
{
    int days = dailyInfected.size();
    vector<int> cumulativeInfected(days, 0), cumulativeRecovered(days, 0), cumulativeFatal(days, 0);
    int totalInf = 0, totalRec = 0, totalFat = 0;
    int firstInfDay = -1, fiftyInfDay = -1, hundredInfDay = -1;

    for (int day = 0; day < days; day++) {
        totalInf += dailyInfected[day];
        totalRec += dailyRecovered[day];
        totalFat += dailyFatal[day];
        cumulativeInfected[day] = totalInf;
        cumulativeRecovered[day] = totalRec;
        cumulativeFatal[day] = totalFat;

        if (firstInfDay == -1 && dailyInfected[day] > 0)
            firstInfDay = day;
        if (fiftyInfDay == -1 && totalInf >= N * 0.5)
            fiftyInfDay = day;
        if (hundredInfDay == -1 && totalInf >= N)
            hundredInfDay = day;
    }

    int firstRecDay = -1, halfRecDay = -1, fullRecDay = -1;
    for (int day = 0; day < days; day++) {
        if (firstRecDay == -1 && dailyRecovered[day] > 0)
            firstRecDay = day;
        if (halfRecDay == -1 && cumulativeRecovered[day] >= totalRec / 2)
            halfRecDay = day;
        if (cumulativeRecovered[day] == totalRec) {
            fullRecDay = day;
            break;
        }
    }

    int firstFatDay = -1, halfFatDay = -1, fullFatDay = -1;
    for (int day = 0; day < days; day++) {
        if (firstFatDay == -1 && dailyFatal[day] > 0)
            firstFatDay = day;
        if (halfFatDay == -1 && cumulativeFatal[day] >= (totalFat + 1) / 2)
            halfFatDay = day;
        if (cumulativeFatal[day] == totalFat) {
            fullFatDay = day;
            break;
        }
    }

    cout << "----- Simulation Results -----" << endl << endl;
    cout << "Total accumulated infected agents: " << totalInf << endl;
    cout << "New infected per day:" << endl;
    for (int day = 0; day < days; day++)
        cout << "  Day " << (day + 1) << ": " << dailyInfected[day]
        << " new, Cumulative: " << cumulativeInfected[day] << endl;
    cout << endl;
    cout << "Day when first infection occurred: " << (firstInfDay == -1 ? "None" : to_string(firstInfDay + 1)) << endl;
    cout << "Day when 50% of the population got infected: " << (fiftyInfDay == -1 ? "Not reached" : to_string(fiftyInfDay + 1)) << endl;
    cout << "Day when 100% of the population got infected: " << (hundredInfDay == -1 ? "Not reached" : to_string(hundredInfDay + 1)) << endl << endl;

    cout << "Total accumulated recovered agents: " << totalRec << endl;
    cout << "New recovered per day:" << endl;
    for (int day = 0; day < days; day++)
        cout << "  Day " << (day + 1) << ": " << dailyRecovered[day]
        << " new, Cumulative: " << cumulativeRecovered[day] << endl;
    cout << endl;
    cout << "Day when first recovery occurred: " << (firstRecDay == -1 ? "None" : to_string(firstRecDay + 1)) << endl;
    cout << "Day when 50% of total recoveries were achieved: " << (halfRecDay == -1 ? "Not reached" : to_string(halfRecDay + 1)) << endl;
    cout << "Day when all recoveries were achieved: " << (fullRecDay == -1 ? "Not reached" : to_string(fullRecDay + 1)) << endl << endl;

    cout << "Total accumulated fatal cases: " << totalFat << endl;
    cout << "New fatal cases per day:" << endl;
    for (int day = 0; day < days; day++)
        cout << "  Day " << (day + 1) << ": " << dailyFatal[day]
        << " new, Cumulative: " << cumulativeFatal[day] << endl;
    cout << endl;
    cout << "Day when first fatal case occurred: " << (firstFatDay == -1 ? "None" : to_string(firstFatDay + 1)) << endl;
    cout << "Day when 50% of total fatal cases were reached: " << (halfFatDay == -1 ? "Not reached" : to_string(halfFatDay + 1)) << endl;
    cout << "Day when all fatal cases were reached: " << (fullFatDay == -1 ? "Not reached" : to_string(fullFatDay + 1)) << endl << endl;

    cout << "CPU Execution Time: " << cpuTime << " seconds" << endl;
    cout << "GPU Execution Time: " << gpuTime << " seconds" << endl;
}

// -----------------------
// Main Function
// -----------------------
int main() {
    // -------- CPU Simulation --------
    Agent* agentesCPU = new Agent[N];
    initializedAgents(agentesCPU);

    vector<int> dailyInfectedCPU, dailyRecoveredCPU, dailyFatalCPU;
    auto startCPU = chrono::high_resolution_clock::now();
    simularCPU(agentesCPU, dailyInfectedCPU, dailyRecoveredCPU, dailyFatalCPU);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedCPU = endCPU - startCPU;

    // -------- GPU Simulation --------
    Agent* agentesGPU;
    curandState* estadosGPU;
    cudaMalloc(&agentesGPU, N * sizeof(Agent));
    cudaMalloc(&estadosGPU, N * sizeof(curandState));

    // Initialize GPU agents using our kernel
    dim3 bloques((N + 255) / 256);
    dim3 hilos(256);
    kernelInicializar << <bloques, hilos >> > (agentesGPU, estadosGPU);
    cudaDeviceSynchronize();

    vector<int> dailyInfectedGPU, dailyRecoveredGPU, dailyFatalGPU;
    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);
    cudaEventRecord(startGPU);
    simularGPU(agentesGPU, estadosGPU, dailyInfectedGPU, dailyRecoveredGPU, dailyFatalGPU);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);
    float elapsedGPU;
    cudaEventElapsedTime(&elapsedGPU, startGPU, endGPU);
    elapsedGPU /= 1000.0f; // Convert milliseconds to seconds

    // -------- Report Results --------
    cout << "----- GPU Simulation Stats -----" << endl;
    printSimulationStats(dailyInfectedGPU, dailyRecoveredGPU, dailyFatalGPU, elapsedCPU.count(), elapsedGPU);
    cout << endl;

    cout << "----- CPU Simulation Stats -----" << endl;
    printSimulationStats(dailyInfectedCPU, dailyRecoveredCPU, dailyFatalCPU, elapsedCPU.count(), elapsedGPU);

    // Free memory
    delete[] agentesCPU;
    cudaFree(agentesGPU);
    cudaFree(estadosGPU);

    return 0;
}
