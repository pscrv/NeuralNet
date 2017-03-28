using System.Collections.Generic;
using Accord.IO;

using NeuralNet;
using System.IO;
using System;

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
        public IEnumerable<VectorPair> TrainingData { get { return _arrayToPairs(_trainingArray); } }
        public IEnumerable<VectorPair> ValidationData { get { return _arrayToPairs(_validationArray); } }
        public IEnumerable<VectorPair> TestData { get { return _arrayToPairs(_testArray); } }

        public TrainingCollection TrainingDataCollection { get { return new TrainingCollection(TrainingData); } }
        public TrainingCollection ValidationDataCollection { get { return new TrainingCollection(ValidationData); } }
        public TrainingCollection TestDataCollection { get { return new TrainingCollection(TestData); } }
        #region public methods
        #endregion

        #region private methods
        private IEnumerable<VectorPair> _arrayToPairs(int[,] array)
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

        private IEnumerable<VectorPair> _arrayToPairs2(int[,] array)
        {
            int nGramCount = array.GetLength(1);
            int lengthMinus1 = array.GetLength(0) - 1;

            //TODO: working here
            for (int i = 0; i < nGramCount; i++)
            {
                Vector inputWord1 = new UnitVector(array[0, i], _vocabularySize);
                Vector inputWord2 = new UnitVector(array[2, i], _vocabularySize);
                Vector inputWord3 = new UnitVector(array[3, i], _vocabularySize);
                Vector outputWord1 = new UnitVector(array[4, i], _vocabularySize);
                yield return new VectorPair(new NetworkVector(inputWord1.ToArray()), new NetworkVector(outputWord1.ToArray()));
            }

            
        }

        #endregion
    }

}
