using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using FourthWord;
using Accord.IO;

namespace FourthWordTests
{
    [TestClass]
    public class Stuff
    {
        [TestMethod]
        public void TestStuff()
        {
            DataReader reader = new DataReader();

            var x = reader.TrainingData;
            var y = reader.TrainingDataCollection;
            y.Add(new NeuralNet.VectorPair(new NeuralNet.NetworkVector(1), new NeuralNet.NetworkVector(1)));
            foreach (var item in y)
            {

            }

        }
    }
}
