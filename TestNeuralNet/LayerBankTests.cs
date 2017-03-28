using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class LayerBankTests
    {
        [TestMethod]
        public void CanMakeLayerBank()
        {
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }));
            LayerBank bank = new LayerBank(layer, 1);
            Assert.AreNotEqual(null, bank);
        }

        [TestMethod]
        public void CannotMakeNullLayerBank()
        {
            Layer layer = null;

            try
            {
                LayerBank bank = new LayerBank(layer, 1);
                Assert.Fail("Failure to throw an NullReferenceException when creating a LayerBank with a null layer.");
            }
            catch (NullReferenceException) { }
        }

        [TestMethod]
        public void CannotMakeZeroLayerBank()
        {
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );

            try
            {
                LayerBank bank = new LayerBank(layer, 0);
                Assert.Fail("Failure to throw an ArgumentException when creating a LayerBank with a < 1 banks.");
            }
            catch (ArgumentException) { }
        }
        
        [TestMethod]
        public void LayerBankHasCorrectOutputSize()
        {
            Layer layer1 = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            Layer layer2 = new Layer (new WeightsMatrix( new double[,] { { 1 }, { 2 }, { 3 } } ) );

            LayerBank bank1a = new LayerBank(layer1, 1);
            LayerBank bank1b = new LayerBank(layer1, 3);
            LayerBank bank2a = new LayerBank(layer2, 2);
            LayerBank bank2b = new LayerBank(layer2, 5);

            Assert.AreEqual(1, bank1a.NumberOfOutputs);
            Assert.AreEqual(bank1a.NumberOfOutputs, bank1a.Output.Dimension);

            Assert.AreEqual(3, bank1b.NumberOfOutputs);
            Assert.AreEqual(bank1b.NumberOfOutputs, bank1b.Output.Dimension);

            Assert.AreEqual(6, bank2a.NumberOfOutputs);
            Assert.AreEqual(bank2a.NumberOfOutputs, bank2a.Output.Dimension);

            Assert.AreEqual(15, bank2b.NumberOfOutputs);
            Assert.AreEqual(bank2b.NumberOfOutputs, bank2b.Output.Dimension);
        }

        [TestMethod]
        public void UnRunLayerBankHasZeroOutput()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector outputcheck = new NetworkVector(numberOfBanks);
            Assert.AreEqual(outputcheck, bank.Output);
        }
        
        [TestMethod]
        public void CanRunLayerBank()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector zeroVector = new NetworkVector(numberOfBanks);

            bank.Run(zeroVector);
            Assert.AreEqual(zeroVector, bank.Output);
        }

        [TestMethod]
        public void CannotRunLayerBankWithBadInputSize()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector badInput = new NetworkVector(2);

            try
            {
                bank.Run(badInput);
                Assert.Fail("LayerBank.Run failed to throw an ArgumentException for input of the wrong size.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void CorrectRunOneInput()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector input = new NetworkVector(new double[] { 2, 3, 5 });
            NetworkVector outputcheck = new NetworkVector(new double[] { 2, 3, 5 });

            bank.Run(input);
            Assert.AreEqual(outputcheck, bank.Output);
        }

        [TestMethod]
        public void CorrectRunThreeInputs()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 2, 3, 5 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector input = new NetworkVector(new double[] { 7, 7, 7, 11, 11, 11, 13, 13, 13 });
            NetworkVector outputcheck = new NetworkVector(new double[] { 70, 110, 130 });

            bank.Run(input);
            Assert.AreEqual(outputcheck, bank.Output);
        }

        [TestMethod]
        public void InputGradient()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector zeroVector = new NetworkVector(numberOfBanks);
            
            Assert.AreEqual(zeroVector, bank.InputGradient(zeroVector));
        }

        [TestMethod]
        public void CannotBackPropagateLayerBankWithBadOutputGradientSize()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector badGradient = new NetworkVector(2);

            try
            {
                bank.BackPropagate(badGradient);
                Assert.Fail("LayerBank.BackPropagate failed to throw an ArgumentException for outputgradient of the wrong size.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void CorrectInputGradientOneInput()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix( new double[,] { { 1 } }) );
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector input = new NetworkVector(new double[] { 2, 3, 5 });
            NetworkVector outputGradient = new NetworkVector(new double[] { 2, 3, 5 });
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 2, 3, 5 });
            
            Assert.AreEqual(inputGradientCheck, bank.InputGradient(outputGradient));
        }

        [TestMethod]
        public void BackPropagateIsCorrect()
        {
            int numberOfBanks = 3;
            Layer layer = new Layer(new WeightsMatrix(new double[,] { { 1 } }));
            LayerBank bank = new LayerBank(layer, numberOfBanks);
            NetworkVector inputVector = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector bpVector = new NetworkVector(new double[] { 5, 7, 11 });

            bank.Run(inputVector);
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 5, 7, 11 });
            Assert.AreEqual(inputGradientCheck, bank.InputGradient(bpVector));


            bank.BackPropagate(bpVector);
            bank.Update(new GradientDescent());

            WeightsMatrix weightsCheck = new WeightsMatrix(new double[,] { { -51 } });
            NetworkVector biasCheck = new NetworkVector(new double[] { -23 });

            Assert.AreEqual(biasCheck, bank.Biases);
            Assert.AreEqual(weightsCheck, bank.Weights);
        }

    }
}
