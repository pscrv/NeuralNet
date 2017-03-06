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
        
        // next line will not be needed, as methods are moved to this class
        // but perhaps needed as _matrix.clone()?
        public double[,] Values { get { return (double[,])_matrix; } }
        #endregion


        #region constructors
        public NetworkMatrix(double[,] matrix)
        {
            _matrix = (double[,])matrix.Clone();
        }
        #endregion


        #region public methods
        public NetworkVector LeftMultiply(NetworkVector vector)
        {
            double sum;
            double[] result = new double[NumberOfNeurons];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                sum = 0;
                for (int j = 0; j < vector.Dimension; j++)
                {
                    sum += _matrix[i, j] * vector.Values[j];
                }
                result[i] = sum;
            }

            return new NetworkVector(result);
        }

        public NetworkVector NeuronWiseWeightAndSum(NetworkVector vector)
        {
            double[] result = new double[NumberOfInputs];
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    result[j] += this._matrix[i, j] * vector.Values[i];
                }
            }

            return new NetworkVector(result);
        }

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
        #endregion



    }
}
