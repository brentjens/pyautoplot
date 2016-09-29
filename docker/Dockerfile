#
# base
#
FROM ubuntu:16.04

#
# common-environment
#
ENV USER=lofar
ENV INSTALLDIR=/opt

#
# environment
#
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.5

#
# versions
#
ENV CASACORE_VERSION=2.1.0 \
    CASAREST_VERSION=1.4.1 \
    PYTHON_CASACORE_VERSION=2.1.2 \
    BOOST_VERSION=1.58

#
# set-uid
#
ENV UID=1000

#
# set-build-options
#
ENV J=8

#
# Base and runtime dependencies
#
#RUN sed -i 's/archive.ubuntu.com/osmirror.rug.nl/' /etc/apt/sources.list 
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y sudo rsync ssh python3 libpython3.5-dev && \
    apt-get install -y libblas3 liblapacke python3-numpy libcfitsio-bin libcfitsio-dev libwcs5 libfftw3-bin libhdf5-10 libboost-python${BOOST_VERSION}.0


#
# open security holes (allow smooth user switching, allow sudo)
#
RUN echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sed -i 's/requiretty/!requiretty/g' /etc/sudoers && \
    chmod a+rw /etc/group /etc/passwd

#
# setup install dir
#
RUN mkdir -p ${INSTALLDIR}

#
# *******************
#   Casacore
# *******************
#
RUN apt-get update && apt-get install -y wget git cmake g++ gfortran flex bison libblas-dev liblapacke-dev libfftw3-dev libhdf5-dev libboost-python-dev libcfitsio3-dev wcslib-dev && \
    mkdir -p ${INSTALLDIR}/casacore/build && \
    mkdir -p ${INSTALLDIR}/casacore/data && \
    cd ${INSTALLDIR}/casacore && git clone https://github.com/casacore/casacore.git src && \
    if [ "${CASACORE_VERSION}" != "latest" ]; then cd ${INSTALLDIR}/casacore/src && git checkout tags/v${CASACORE_VERSION}; fi && \
    cd ${INSTALLDIR}/casacore/data && wget --retry-connrefused ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
    cd ${INSTALLDIR}/casacore/data && tar xf WSRT_Measures.ztar  && rm -f WSRT_Measures.ztar && \
    cd ${INSTALLDIR}/casacore/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/casacore/ -DDATA_DIR=${INSTALLDIR}/casacore/data -DBUILD_PYTHON=False -DBUILD_PYTHON3=True -DPYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython${PYTHON_VERSION}m.so -DPYTHON3_INCLUDE_DIR=/usr/include/python${PYTHON_VERSION}  -DUSE_OPENMP=True -DUSE_FFTW3=TRUE ../src/ && \
    cd ${INSTALLDIR}/casacore/build && make -j ${J} && \
    cd ${INSTALLDIR}/casacore/build && make install && \
    bash -c "strip ${INSTALLDIR}/casacore/{lib,bin}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/casacore/{build,src}" && \
    apt-get purge -y wget git cmake g++ gfortran flex bison libblas-dev liblapacke-dev libfftw3-dev libhdf5-dev libboost-python-dev libcfitsio3-dev wcslib-dev && \
    apt-get autoremove -y

#
# *******************
#   Casarest
# *******************
#
RUN apt-get update && apt-get install -y git cmake g++ gfortran libboost-system-dev libboost-thread-dev libhdf5-dev libcfitsio3-dev wcslib-dev libblas-dev liblapacke-dev && \
    mkdir -p ${INSTALLDIR}/casarest/build && \
    cd ${INSTALLDIR}/casarest && git clone https://github.com/casacore/casarest.git src && \
    if [ "${CASAREST_VERSION}" != "latest" ]; then cd ${INSTALLDIR}/casarest/src && git checkout tags/v${CASAREST_VERSION}; fi && \
    cd ${INSTALLDIR}/casarest/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/casarest -DCASACORE_ROOT_DIR=${INSTALLDIR}/casacore ../src/ && \
    cd ${INSTALLDIR}/casarest/build && make -j ${J} && \
    cd ${INSTALLDIR}/casarest/build && make install && \
    bash -c "strip ${INSTALLDIR}/casarest/{lib,bin}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/casarest/{build,src}" && \
    apt-get purge -y git cmake g++ gfortran libboost-system-dev libboost-thread-dev libhdf5-dev libcfitsio3-dev wcslib-dev libblas-dev liblapacke-dev && \
    apt-get autoremove -y

#
# *******************
#   Pyrap
# *******************
#
RUN apt-get update && apt-get install -y git make g++ python3-setuptools libboost-python-dev libcfitsio3-dev wcslib-dev && \
    mkdir ${INSTALLDIR}/python-casacore && \
    cd ${INSTALLDIR}/python-casacore && git clone https://github.com/casacore/python-casacore && \
    if [ "$PYTHON_CASACORE_VERSION" != "latest" ]; then cd ${INSTALLDIR}/python-casacore/python-casacore && git checkout tags/v${PYTHON_CASACORE_VERSION}; fi && \
    cd ${INSTALLDIR}/python-casacore/python-casacore && python3 ./setup.py build_ext -I${INSTALLDIR}/casacore/include/ -L${INSTALLDIR}/casacore/lib/ && \
    mkdir -p ${INSTALLDIR}/python-casacore/lib/python${PYTHON_VERSION}/site-packages/ && \
    mkdir -p ${INSTALLDIR}/python-casacore/lib64/python${PYTHON_VERSION}/site-packages/ && \
    export PYTHONPATH=${INSTALLDIR}/python-casacore/lib/python${PYTHON_VERSION}/site-packages:${INSTALLDIR}/python-casacore/lib64/python${PYTHON_VERSION}/site-packages:$PYTHONPATH && cd ${INSTALLDIR}/python-casacore/python-casacore && python3 ./setup.py install --prefix=${INSTALLDIR}/python-casacore/ && \
    bash -c "find ${INSTALLDIR}/python-casacore/lib -name '*.so' | xargs strip || true" && \
    bash -c "rm -rf ${INSTALLDIR}/python-casacore/python-casacore" && \
    apt-get purge -y git make g++ python-setuptools libboost-python-dev libcfitsio3-dev wcslib-dev && \
    apt-get autoremove -y

#
# *******************
#   QPID client
# *******************
#

# Run-time dependencies
# QPID daemon legacy store would require: libaio1 libdb5.1++
##RUN apt-get update && apt-get install -y sasl2-bin libuuid1 libnss3 libnspr4 xqilla libboost-program-options${BOOST_VERSION}.0 libboost-filesystem${BOOST_VERSION}.0

# Install
# QPID daemon legacy store would require: libaio-dev libdb5.1++-dev
##RUN apt-get update && apt-get install -y subversion swig ruby ruby-dev python-dev libsasl2-dev pkg-config cmake libtool uuid-dev libxerces-c-dev libnss3-dev libnspr4-dev help2man fakeroot build-essential debhelper libsslcommon2-dev libxqilla-dev python-setuptools libboost-program-options${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev && \
##    mkdir /opt/qpid && \
##    svn --non-interactive -q co https://svn.astron.nl/LOFAR/trunk/LCS/MessageBus/qpid/ /opt/qpid; \
##    /opt/qpid/local/sbin/build_qpid && \
##    bash -c "strip /opt/qpid/{bin,lib}/* || true" && \
##    bash -c "rm -rf ~/sources" && \
##    apt-get purge -y subversion swig ruby ruby-dev python-dev libsasl2-dev pkg-config cmake libtool uuid-dev libxerces-c-dev libnss3-dev libnspr4-dev help2man fakeroot build-essential debhelper libsslcommon2-dev libxqilla-dev python-setuptools libboost-program-options${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev && \
##    apt-get autoremove -y

#
# *******************
#   DAL
# *******************
#

# RUN apt-get update &&&& apt-get install -y git cmake g++ swig python-dev libhdf5-dev && \
#     mkdir ${INSTALLDIR}/DAL && \
#     cd ${INSTALLDIR}/DAL && git clone https://github.com/nextgen-astrodata/DAL.git src && \
#     mkdir ${INSTALLDIR}/DAL/build && cd ${INSTALLDIR}/DAL/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/DAL ../src && \
#     make -j ${J} && \
#     make install && \
#     bash -c "find ${INSTALLDIR}/DAL/lib -name '*.so' | xargs strip || true" && \
#     bash -c "rm -rf ${INSTALLDIR}/DAL/{src,build}" && \
#     apt-get purge -y git cmake g++ swig python-dev libhdf5-dev && \
#     apt-get autoremove -y

#
# *******************
#   LofarStMan
# *******************
#

# Tell image build information
ENV LOFAR_BRANCH_URL=https://svn.astron.nl/LOFAR/trunk \
    LOFAR_REVISION=HEAD

# Install
RUN apt-get update && apt-get install -y subversion cmake g++ libboost-dev && \
    mkdir -p ${INSTALLDIR}/lofar/build/gnu_opt && \
    cd ${INSTALLDIR}/lofar && \
    svn --non-interactive -q co -r ${LOFAR_REVISION} -N ${LOFAR_BRANCH_URL} src; \
    svn --non-interactive -q up src/CMake && \
    cd ${INSTALLDIR}/lofar/build/gnu_opt && cmake -DBUILD_PACKAGES=LofarStMan -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/lofar/ -DCASACORE_ROOT_DIR=${INSTALLDIR}/casacore/ -DUSE_LOG4CPLUS=OFF -DUSE_OPENMP=True ${INSTALLDIR}/lofar/src/ && \
    cd ${INSTALLDIR}/lofar/build/gnu_opt && make -j ${J} && \
    cd ${INSTALLDIR}/lofar/build/gnu_opt && make install && \
    bash -c "strip ${INSTALLDIR}/lofar/{bin,sbin,lib64}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/lofar/{build,src}" && \
    apt-get purge -y subversion cmake g++ libboost-dev && \
    apt-get autoremove -y

#
#**************************
#    PYAUTOPLOT
#**************************
#

RUN apt-get update -y && \
    apt-get install -y python3-numpy python3-matplotlib python3-scipy python3-setuptools \
                          wget git  python3-nose python3-coverage \
                          python3-astropy

RUN A=$F; git clone https://github.com/ruudiculus/pyautoplot.git ${INSTALLDIR}/pyautoplot && \
    cd ${INSTALLDIR}/pyautoplot && \
    python3 setup.py install


#
# entry
#
COPY ["bashrc",    "${INSTALLDIR}/"]
COPY ["bashrc.d",  "${INSTALLDIR}/bashrc.d/"]
COPY ["chuser.sh", "/usr/local/bin"]
ENTRYPOINT ["/usr/local/bin/chuser.sh"]

