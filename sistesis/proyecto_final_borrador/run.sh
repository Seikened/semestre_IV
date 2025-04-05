clear
echo "Ejecutando el script run.sh"
pip uninstall qiskit qiskit-terra qiskit-aer qiskit-algorithms -y
echo "Instalando dependencias"
pip install -r requirements.txt                                                                                       