using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using System.Collections.Generic;

namespace TestNeuralNet
{
    [TestClass]
    public class UnitNetworkVectorTests
    {
        [TestMethod]
        public void CanMake()
        {
            Vector vector = new UnitVector(0, 1);
            Assert.IsNotNull(vector);
        }

        [TestMethod]
        public void CanRead()
        {
            Vector vector = new UnitVector(1, 2);
            Assert.AreEqual(vector[0], 0.0);
            Assert.AreEqual(vector[1], 1.0);
        }

        [TestMethod]
        public void CannotMakeDimensionLessThan1()
        {
            try
            {
                Vector vector = new UnitVector(0, 0);
                Assert.Fail("Should throw an ArgumentException, but did not.");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void CannotMakeWithBadIndex()
        {
            try
            {
                Vector vector = new UnitVector(-1, 2);
                Assert.Fail("Should throw an IndexOutOfRangeException, but did not.");
            }
            catch (IndexOutOfRangeException)
            { }

            try
            {
                Vector vector = new UnitVector(2, 2);
                Assert.Fail("Should throw an IndexOutOfRangeException, but did not.");
            }
            catch (IndexOutOfRangeException)
            { }
        }
        
        [TestMethod]
        public void CannotZero()
        {
            NetworkVector vector = new UnitNetworkVector(1, 2);

            try
            {
                vector.Zero();
                Assert.Fail("Failure to throw an InvalidOperationException.");
            }
            catch (InvalidOperationException) { }
        }

        [TestMethod]
        public void CanCompareEquality()
        {
            NetworkVector vector1 = new UnitNetworkVector(0, 3);
            NetworkVector vector2 = new UnitNetworkVector(0, 3);
            NetworkVector vector3 = new UnitNetworkVector(0, 2);
            bool areEqual1 = vector1.Equals(vector2);
            bool areEqual2 = vector1.Equals(vector1);
            Assert.IsTrue(areEqual1);
            Assert.IsTrue(areEqual2);
            Assert.IsFalse(vector1.Equals(vector3));
            Assert.IsFalse(vector3.Equals(vector1));
            Assert.IsFalse(vector1.Equals(null));
            Assert.IsFalse(vector1.Equals(new WeightsMatrix(3, 5)));
        }

        [TestMethod]
        public void TestHash()
        {
            NetworkVector vector1 = new UnitNetworkVector(0, 2);
            NetworkVector vector2 = new UnitNetworkVector(0, 2);
            NetworkVector vector3 = new UnitNetworkVector(0, 1);
            Assert.AreEqual(vector1.GetHashCode(), vector2.GetHashCode());
            Assert.AreNotEqual(vector1.GetHashCode(), vector3.GetHashCode());
        }
        

        [TestMethod]
        public void CanMakeAndReadBigUnitNetworkVector()
        {
            int size = 10000;
            int index = 10;
            NetworkVector vector = new UnitNetworkVector(index, size);
            double[] array = vector.ToArray();

            Assert.AreEqual(array[10], 1.0);
            for (int i = 11; i < size; i++)
                Assert.AreEqual(array[i], 0.0);
        }


        [TestMethod]
        public void CanApplyFunctionComponentwise()
        {
            NetworkVector.SingleVariableFunction f = x => x - 1;
            NetworkVector vector = new UnitNetworkVector(1, 2);
            NetworkVector newvector = NetworkVector.ApplyFunctionComponentWise(vector, f);
            double[] vectorValues = newvector.ToArray();
            Assert.AreEqual(-1.0, vectorValues[0]);
            Assert.AreEqual(0.0, vectorValues[1]);
        }


        [TestMethod]
        public void CannotSubtract()
        {
            NetworkVector vector = new UnitNetworkVector(0, 5);

            try
            {
                vector.Subtract(vector);
                Assert.Fail("Failure to throw and InvalidOperationException.");
            }
            catch (InvalidOperationException) { }

        }

        [TestMethod]
        public void CanSum()
        {
            int listSize = 1000000;
            NetworkVector vector = new UnitNetworkVector(0, 5);
            List<NetworkVector> list = new List<NetworkVector>();

            for (int i = 0; i < listSize; i++)
            {
                list.Add(vector);
            }

            vector = NetworkVector.Sum(list);
            double[] result = vector.ToArray();

            Assert.AreEqual(listSize, result[0]);
            for (int i = 1; i < vector.Dimension; i++)
            {
                Assert.AreEqual(0, result[i]);
            }
        }


        [TestMethod]
        public void CanMakeDotProduct()
        {
            NetworkVector vector = new UnitNetworkVector(0, 3);
            double dot = vector.DotProduct(vector);
            Assert.AreEqual(1, dot);
        }


        [TestMethod]
        public void CanMultiply()
        {
            NetworkVector vector1 = new UnitNetworkVector(0, 2);
            NetworkVector vector2 = new UnitNetworkVector(0, 2);
            WeightsMatrix product = vector1.LeftMultiply(vector2);
            double[,] productValues = product.ToArray();
            Assert.AreEqual(1, productValues[0, 0]);
            Assert.AreEqual(0, productValues[0, 1]);
            Assert.AreEqual(0, productValues[1, 0]);
            Assert.AreEqual(0, productValues[1, 1]);
        }

        [TestMethod]
        public void CanCopy()
        {
            NetworkVector vector1 = new UnitNetworkVector(0, 3);
            NetworkVector vector2 = vector1.Copy();
            double[] vector1Values = vector1.ToArray();
            double[] vector2Values = vector2.ToArray();
            Assert.AreEqual(vector1Values[0], vector2Values[0]);
            Assert.AreEqual(vector1Values[1], vector2Values[1]);
        }
    }
    
}
