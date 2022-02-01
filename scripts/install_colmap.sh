git clone https://github.com/ceres-solver/ceres-solver.git -b 2.0.0 && \
    cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j8 && \
    make install

git clone https://github.com/colmap/colmap.git -b 3.7 && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install