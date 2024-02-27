# Import dolfinX-Image
FROM dolfinx/dolfinx:stable

# --- Modify docker container ---
# Update docker container
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade &&\
    apt-get -y install vim &&\
    apt-get -y install libboost-program-options-dev

# Add bash_profil
RUN touch "/root/.bash_aliases"
RUN echo "alias ..='cd ..'" >> "/root/.bash_aliases"

# Some additional python packages
RUN pip3 install matplotlib numexpr sympy

# --- Add dolfinx_poro to container ---
# Create folder structure
RUN mkdir -p /root/Programs/dolfinx_poro
RUN mkdir -p /root/Programs/dolfinx_poro/dolfinx_poro

# Add source code
ADD dolfinx_poro/dolfinx_poro /root/Programs/dolfinx_poro/dolfinx_poro
ADD dolfinx_poro/setup.py /root/Programs/dolfinx_poro

# Installation
RUN pip3 install /root/Programs/dolfinx_poro/ -v --upgrade