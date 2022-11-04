
# .bash_logout sometimes causes a spurious bad exit code, remove it.
rm ~/.bash_logout
rm -Rf jax

echo "Clean up" 
pip uninstall jax jaxlib libtpu-nightly libtpu -y 
          
# Via https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source
pip install numpy six wheel
          
echo "Checking out and installing JAX..."
git clone https://github.com/yejingxin/jax.git
cd jax
git remote add upstream https://github.com/google/jax.git
git fetch upstream
git merge upstream/main
echo "jax git hash: $(git rev-parse HEAD)"
pip install -r build/test-requirements.txt
pip install -e .[tpu] -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
          
python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
          
echo "Installing latest libtpu-nightly..."
          
python3 -c 'import jax; print("jax version:", jax.__version__)'
python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
python3 -c 'import jax; print("libtpu version:",
  jax.lib.xla_bridge.get_backend().platform_version)'