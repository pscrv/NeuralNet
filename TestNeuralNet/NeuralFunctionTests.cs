﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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

        NeuralFunction nf_1 = new NeuralFunction(1);

        NeuralFunction nf_linear = 
            new NeuralFunction(2, x => x, (x, y) => 1);

        NeuralFunction nf_sigmoid 
            = new NeuralFunction(2, sigmoid, sigmoidDerivative);

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
            NetworkVector result = nf_1.Run(vector_1);
            Assert.AreEqual(vector_1, result);
        }

        [TestMethod]
        public void CanRunLinear()
        {
            NetworkVector result = nf_linear.Run(vector_2);
            Assert.AreEqual(vector_2, result);
        }

        [TestMethod]
        public void CanRunSigmoid()
        {
            NetworkVector result = nf_sigmoid.Run(vector_2);
            NetworkVector outputCheck = new NetworkVector(new double[] { sigmoid(1), sigmoid(2) });
            Assert.AreEqual(outputCheck, result);
        }

        [TestMethod]
        public void CanBack()
        {
            nf_1.Run(vector_1);
            Assert.AreEqual(vector_1, nf_1.InputGradient(vector_1));
        }

        [TestMethod]
        public void CanBackLinear()
        {
            NetworkVector result = nf_linear.Run(vector_2);
            NetworkVector inputgradienttest = new NetworkVector(new double[] { 1, 1 });
            Assert.AreEqual(inputgradienttest, nf_linear.InputGradient(vector_2, vector_2, result));
        }

        [TestMethod]
        public void CanBackSigmoid()
        {
            NetworkVector result = nf_sigmoid.Run(vector_2);
            double[] outarray = result.ToArray();
            double[] inarray = vector_2.ToArray();
            NetworkVector gradientCheck = new NetworkVector(
                new double[] { sigmoidDerivative(inarray[0], outarray[0]),
                    sigmoidDerivative(inarray[1], outarray[1]) });
            Assert.AreEqual(gradientCheck, nf_sigmoid.InputGradient(vector_2, vector_2, result));
        }
    }
}
