using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using System.Collections.Generic;

namespace TestNeuralNet
{
    [TestClass]
    public class NetworkVectorTests
    {
        [TestMethod]
        public void CanMakeAndReadZeroNetworkVector()
        {
            NetworkVector vector = new NetworkVector(2);
            Assert.AreEqual(0.0, vector.ToArray()[0]);
            Assert.AreEqual(0.0, vector.ToArray()[1]);
        }

        [TestMethod]
        public void CanCompareEquality()
        {
            NetworkVector vector1 = new NetworkVector(2);
            NetworkVector vector2 = new NetworkVector(new double[] { 0, 0 });
            NetworkVector vector3 = new NetworkVector(3);
            bool areEqual1 = vector1.Equals(vector2);
            bool areEqual2 = vector1.Equals(vector1);
            Assert.IsTrue(areEqual1);
            Assert.IsTrue(areEqual2);
            Assert.IsFalse(vector1.Equals(vector3));
            Assert.IsFalse(vector3.Equals(vector1));
            Assert.IsFalse(vector1.Equals(null));
            Assert.IsFalse(vector1.Equals(new NetworkMatrix(3, 5)));
        }

        [TestMethod]
        public void TestHash()
        {
            NetworkVector vector1 = new NetworkVector(2);
            NetworkVector vector2 = new NetworkVector(new double[] { 0, 0 });
            NetworkVector vector3 = new NetworkVector(3);
            Assert.AreEqual(vector1.GetHashCode(), vector2.GetHashCode());
            Assert.AreNotEqual(vector1.GetHashCode(), vector3.GetHashCode());
        }

        [TestMethod]
        public void CanMakeAndRead_1_2_NetworkVector()
        {
            NetworkVector vector = new NetworkVector(new double[] { 1, 2 });
            double[] vectorValues = vector.ToArray();
            Assert.AreEqual(1.0, vectorValues[0]);
            Assert.AreEqual(2.0, vectorValues[1]);
        }

        [TestMethod]
        public void CanMakeAndReadBigNetworkVector()
        {
            int size = 10000;
            double[] array = new double[size];
            for (int i = 0; i < size; i++)
            {
                array[i] = i;
            }
            NetworkVector vector = new NetworkVector(array);
            double[] vectorValues = vector.ToArray();
            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual(array[i], vectorValues[i]);
            }
        }


        [TestMethod]
        public void CanApplyFunctionComponentwise()
        {
            NetworkVector.SingleVariableFunction f = x => x - 1;
            NetworkVector vector = new NetworkVector(new double[] { 1, 2 });
            NetworkVector newvector = NetworkVector.ApplyFunctionComponentWise(vector, f);
            double[] vectorValues = newvector.ToArray();
            Assert.AreEqual(0.0, vectorValues[0]);
            Assert.AreEqual(1.0, vectorValues[1]);
        }
        

        [TestMethod]
        public void CanSubtract()
        {
            NetworkVector vector = new NetworkVector(new double[] { 1, 2, 3, 4, 5 });
            vector.Subtract(vector);
            double[] vectorValues = vector.ToArray();
            for (int i = 0; i < vectorValues.Length; i++)
            {
                Assert.AreEqual(0.0, vectorValues[i]);
            }
        }

        [TestMethod]
        public void CanSum()
        {
            int listSize = 1000000;
            NetworkVector vector = new NetworkVector(new double[] { 0, 1, 2, 3, 4, 5 });
            List<NetworkVector> list = new List<NetworkVector>();

            for (int i = 0; i < listSize; i++)
            {
                list.Add(vector);
            }

            vector = NetworkVector.Sum(list);
            double[] result = vector.ToArray();
            for (int i = 0; i < vector.Dimension; i++)
            {
                Assert.AreEqual(i * listSize, result[i]);
            }
        }


        [TestMethod]
        public void CanMakeDotProduct()
        {
            NetworkVector vector = new NetworkVector(new double[] { 1, 2, 3, 4, 5 });
            double dot = vector.DotProduct(vector);
            Assert.AreEqual(55, dot);
        }


        [TestMethod]
        public void CanMultiply()
        {
            NetworkVector vector1 = new NetworkVector(new double[] { 1, 0 });
            NetworkVector vector2 = new NetworkVector(new double[] { 0, 1 });
            NetworkMatrix product = vector1.LeftMultiply(vector2);
            double[,] productValues = product.ToArray();
            Assert.AreEqual(0, productValues[0, 0]);
            Assert.AreEqual(1, productValues[0, 1]);
            Assert.AreEqual(0, productValues[1, 0]);
            Assert.AreEqual(0, productValues[1, 1]);
        }

        [TestMethod]
        public void CanCopy()
        {
            NetworkVector vector1 = new NetworkVector(new double[] { 1, 0 });
            NetworkVector vector2 = vector1.Copy();
            double[] vector1Values = vector1.ToArray();
            double[] vector2Values = vector2.ToArray();
            Assert.AreEqual(vector1Values[0], vector2Values[0]);
            Assert.AreEqual(vector1Values[1], vector2Values[1]);
        }
    }



    [TestClass]
    public class NetworkMatrixTests
    {
        [TestMethod]
        public void CanMakeAndReadZeroNetworkMatrix()
        {
            int neuronCount = 2;
            int inputcount = 3;
            NetworkMatrix matrix = new NetworkMatrix(neuronCount, inputcount);

            double[,] matrixValues = matrix.ToArray();
            Assert.AreEqual(neuronCount, matrix.NumberOfOutputs);
            Assert.AreEqual(inputcount, matrix.NumberOfInputs);
            for (int i = 0; i < neuronCount; i++)
                for (int j = 0; j < inputcount; j++)
                    Assert.AreEqual(0.0, matrixValues[i, j]);
        }

        [TestMethod]
        public void CanMakeAndReadNoneroNetworkMatrix()
        {
            int neuronCount = 2;
            int inputcount = 3;
            NetworkMatrix matrix = new NetworkMatrix(new double[,]{ {0, 1, 2 }, { 1, 2, 3} });

            double[,] matrixValues = matrix.ToArray();
            Assert.AreEqual(neuronCount, matrix.NumberOfOutputs);
            Assert.AreEqual(inputcount, matrix.NumberOfInputs);
            for (int i = 0; i < neuronCount; i++)
                for (int j = 0; j < inputcount; j++)
                    Assert.AreEqual(i + j, matrixValues[i, j]);
        }

        [TestMethod]
        public void CanSubtract()
        {
            int neuronCount = 2;
            int inputcount = 3;
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            matrix.Subtract(matrix);

            double[,] matrixValues = matrix.ToArray();
            for (int i = 0; i < neuronCount; i++)
                for (int j = 0; j < inputcount; j++)
                    Assert.AreEqual(0, matrixValues[i, j]);
        }

        [TestMethod]
        public void CanLeftMultiply()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 0, 1 }, { 1, 0 } });
            NetworkVector vector = new NetworkVector(new double[] { 1, 1 });
            NetworkVector result = matrix.LeftMultiply(vector);

            double[] resultValues = result.ToArray();
            Assert.AreEqual(1, resultValues[0]);
            Assert.AreEqual(1, resultValues[1]);
        }

        [TestMethod]
        public void CanDotWithWeightsPerInput()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 0, 1 }, { 1, 0 } });
            NetworkVector vector = new NetworkVector(new double[] { 1, 0 });
            NetworkVector result = matrix.DotWithWeightsPerInput(vector);

            double[] resultValues = result.ToArray();
            Assert.AreEqual(0, resultValues[0]);
            Assert.AreEqual(1, resultValues[1]);
        }

        //TODO: write tests for equality and hash

    }
}

