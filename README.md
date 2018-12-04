# feynmann_path_integral

At UCSD I completed a project that can run a simulation of the Feynman Path Integral to find the solution for the Quantum Harmonic Oscillator in an unvisualized polynomial potential well, along with a video showing the simulation. The simulation is able to be run and visualized in real time because it is executed on the GPU using Nvidia's CUDA technology with OpenGL interoperability. Interoperability means that the memory locations on the device that are used for computations can be seamlessly read by OpenGL for writing to the screen; therefore, the data does not need to be transfered back and forth from the GPU devices' memory to the main memory on the host computer which would add a significant compute time overhead. I added this feature from studying CUDA and OpenGL out of personal interest, it was not required for the original assignment.

http://www.youtube-nocookie.com/embed/ruPjyH0BFpc?rel=0
