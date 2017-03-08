using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NetworkMatrix
    {
        #region private attributes
        private double[,] _matrix;
        #endregion

        #region public properties
        public int NumberOfNeurons { get { return _matrix.GetLength(0); } }
        public int NumberOfInputs { get { return _matrix.GetLength(1); } }        
        #endregion


        #region constructors
        public NetworkMatrix(double[,] matrix)
        {
            _matrix = (double[,])matrix.Clone();
        }

        public NetworkMatrix(int neurons, int inputs)
        {
            _matrix = new double[neurons, inputs];
        }
        #endregion


        #region public methods
        public void Subtract(NetworkMatrix other)
        {
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    this._matrix[i, j] -= other._matrix[i, j];
                }
            }
        }
        
        public NetworkVector LeftMultiply(NetworkVector vector)
        {
            double[] vectorarray = vector.ToArray();
            double sum;
            double[] result = new double[NumberOfNeurons];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                sum = 0;
                for (int j = 0; j < vector.Dimension; j++)
                {
                    sum += _matrix[i, j] * vectorarray[j];
                }
                result[i] = sum;
            }

            return new NetworkVector(result);
        }
        
        public NetworkVector DotWithWeightsPerInput(NetworkVector vector)
        {
            double[] result = new double[NumberOfInputs];
            NetworkVector inputWeights;
            for (int i = 0; i < NumberOfInputs; i++)
            {
                inputWeights = _getWeightsForOneInput(i);
                result[i] = vector.DotProduct(inputWeights);
            }
            
            return new NetworkVector(result);
        }

        public double[,] ToArray()
        {
            return (double[,]) _matrix.Clone();
        }
        #endregion

        #region private methods
        private NetworkVector _getWeightsForOneInput(int inputindex)
        {
            double[] result = new double[NumberOfNeurons];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                result[i] = _matrix[i, inputindex];
            }
            return new NetworkVector(result);
        }
        #endregion
    }
}
