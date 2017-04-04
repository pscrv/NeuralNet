using System.Collections.Generic;
using Accord.IO;

using NeuralNet;
using System.IO;
using System;
using MathNet.Numerics.LinearAlgebra;

namespace FourthWord
{
    public class DataReader
    {
        #region private attributes
        private string _matlabFile = "C:\\Users\\Paul\\Documents\\NN_Assignments\\assignment2\\data.mat";

        private int[,] _trainingArray;
        private int[,] _validationArray;
        private int[,] _testArray;
        private int _vocabularySize = 250;
        #endregion


        #region constructors
        public DataReader()
        {
            using (MatReader reader = new MatReader(File.OpenRead(_matlabFile)))
            {
                _trainingArray = reader["data"]["trainData"].Value as int[,];
                _validationArray = reader["data"]["validData"].Value as int[,];
                _testArray = reader["data"]["testData"].Value as int[,];
            }
        }
        #endregion
        public IEnumerable<VectorPair> TrainingData { get { return _arrayToVectorPairs(_trainingArray); } }
        public IEnumerable<VectorPair> ValidationData { get { return _arrayToVectorPairs(_validationArray); } }
        public IEnumerable<VectorPair> TestData { get { return _arrayToVectorPairs(_testArray); } }

        public TrainingCollection TrainingDataCollection { get { return new TrainingCollection(TrainingData); } }
        public TrainingCollection ValidationDataCollection { get { return new TrainingCollection(ValidationData); } }
        public TrainingCollection TestDataCollection { get { return new TrainingCollection(TestData); } }

        public TrainingBatchCollection TrainingBatchDataCollection(int batchsize)
        {
            return new TrainingBatchCollection( _arrayToBatchPairs(_trainingArray, batchsize) ); 
        }
        #region public methods
        #endregion

        #region private methods
        private IEnumerable<VectorPair> _arrayToVectorPairs(int[,] array)
        {
            int nGramCount = array.GetLength(1);
            int lengthMinus1 = array.GetLength(0) - 1;
            int inputVectorDimension = lengthMinus1 * _vocabularySize;

            double[] inputVector = new double[inputVectorDimension];
            double[] outputVector = new double[_vocabularySize];
            for (int i = 0; i < nGramCount; i++)
            {
                for (int j = 0; j < lengthMinus1; j++)
                {
                    inputVector[(j * _vocabularySize) + (array[j, i] - 1)] = 1;
                }

                outputVector[array[lengthMinus1, i] - 1] = 1;

                yield return new VectorPair(new NetworkVector(inputVector), new NetworkVector(outputVector));
                Array.Clear(inputVector, 0, inputVector.Length);
                Array.Clear(outputVector, 0, outputVector.Length);
            }
        }
        private IEnumerable<BatchPair> _arrayToBatchPairs(int[,] array, int batchsize)
        {
            int nGramCount = array.GetLength(1);
            int lengthMinus1 = array.GetLength(0) - 1;
            int inputVectorDimension = lengthMinus1 * _vocabularySize;

            Matrix<double> inputBatch = Matrix<double>.Build.Dense(batchsize, inputVectorDimension);
            Matrix<double> outputBatch = Matrix<double>.Build.Dense(batchsize, _vocabularySize);
            
            int count = 0;
            int batchcount = 0;
            for (int i = 0; i < nGramCount; i++)
            {
                for (int j = 0; j < lengthMinus1; j++)
                {
                    inputBatch[i - (batchcount * batchsize), j * _vocabularySize + array[j, i] - 1] = 1;
                }

                outputBatch[i - (batchcount * batchsize), array[lengthMinus1, i] - 1] = 1;

                count++;
                if (count == batchsize)
                {
                    batchcount++;
                    count = 0;
                    yield return new BatchPair(new VectorBatch(inputBatch), new VectorBatch(outputBatch));
                }
            }
        }

        #endregion
    }

}
