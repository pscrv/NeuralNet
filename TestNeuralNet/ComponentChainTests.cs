using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class ComponentChainTests
    {
        [TestMethod]
        public void CanChainWeightCombinerWithSoftMax()
        {
            NetworkComponent layer = new LinearLayer(new double[,] { { 1 } });
            NetworkComponent smu = new SoftMaxUnit(1);

            NetworkComponentChain smlayer = new NetworkComponentChain();
            smlayer.Add(layer);
            smlayer.Add(smu);
        }
    }
}
