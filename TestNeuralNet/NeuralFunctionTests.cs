using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet.NetComponent;
using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class NeuralFunctionTests
    {
        static private double sigmoid(double input)
        {
            return 1.0 / (1 + Math.Exp(-input));
        }

        static private double sigmoidDerivative(double input, double output)
        {
            return output * (1 - output);
        }

        NeuralNet.NetComponent.NeuralFunction nf_1 = new NeuralNet.NetComponent.NeuralFunction(1);

        NeuralNet.NetComponent.NeuralFunction nf_linear = 
            new NeuralNet.NetComponent.NeuralFunction(2, x => x, (x, y) => 1);

        NeuralNet.NetComponent.NeuralFunction nf_sigmoid 
            = new NeuralNet.NetComponent.NeuralFunction(2, sigmoid, sigmoidDerivative);

        NetworkVector vector_1 = new NetworkVector(new double[] { 1 });
        NetworkVector vector_2 = new NetworkVector(new double[] { 1, 2 });


        [TestMethod]
        public void CanMake()
        {
            Assert.IsNotNull(nf_1);
        }

        [TestMethod]
        public void CanMakeLinear()
        {
            Assert.IsNotNull(nf_linear);
        }

        [TestMethod]
        public void CanMakeSigmoid()
        {
            Assert.IsNotNull(nf_sigmoid);
        }

        [TestMethod]
        public void CanRun()
        {
            nf_1.Run(vector_1);
            Assert.AreEqual(vector_1, nf_1.Output);
        }

        [TestMethod]
        public void CanRunLinear()
        {
            nf_linear.Run(vector_2);
            Assert.AreEqual(vector_2, nf_linear.Output);
        }

        [TestMethod]
        public void CanRunSigmoid()
        {
            nf_sigmoid.Run(vector_2);
            NetworkVector outputCheck = new NetworkVector(new double[] { sigmoid(1), sigmoid(2) });
            Assert.AreEqual(outputCheck, nf_sigmoid.Output);
        }

        [TestMethod]
        public void CanBack()
        {
            nf_1.Run(vector_1);
            //nf_1.Back(vector_1);
            Assert.AreEqual(vector_1, nf_1.InputGradient(vector_1));
        }

        [TestMethod]
        public void CanBackLinear()
        {
            nf_linear.Run(vector_2);
            //nf_linear.Back(vector_2);
            NetworkVector inputgradienttest = new NetworkVector(new double[] { 1, 1 });
            Assert.AreEqual(inputgradienttest, nf_linear.InputGradient(vector_2));
        }

        [TestMethod]
        public void CanBackSigmoid()
        {
            nf_sigmoid.Run(vector_2);
            //nf_sigmoid.Back(vector_2);
            double[] outarray = nf_sigmoid.Output.ToArray();
            double[] inarray = vector_2.ToArray();
            NetworkVector outputCheck = new NetworkVector(
                new double[] { sigmoidDerivative(inarray[0], outarray[0]),
                    sigmoidDerivative(inarray[1], outarray[1]) });
            Assert.AreEqual(outputCheck, nf_sigmoid.InputGradient(vector_2));
        }
    }
}
