using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class TrainerTests
    {
        [TestMethod]
        public void CanMakeOnlineTrainer()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunOnlineTrainer()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void TrainCorrectOnePass()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ), 
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
            
            double[,] weightsCheck = new double[,] { { 1, 3 } };
            double[] biasesCheck = new double[] { 1 };

            for (int i = 0; i < layer.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

                for (int j = 0; j < layer.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
                }

            }
        }

        [TestMethod]
        public void TrainCorrectTwoPasses()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
            trainer.Train();

            double[,] weightsCheck = new double[,] { { 3, 5 } };
            double[] biasesCheck = new double[] { 1 };

            for (int i = 0; i < layer.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

                for (int j = 0; j < layer.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
                }

            }
        }
        
        [TestMethod]
        public void TrainCorrectThreePasses()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
            trainer.Train();
            trainer.Train();

            double[,] weightsCheck = new double[,] { { 3, 7 } };
            double[] biasesCheck = new double[] { -1 };

            for (int i = 0; i < layer.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

                for (int j = 0; j < layer.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
                }

            }
        }



        [TestMethod]
        public void CanMakeBatchTrainer()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } });
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunBatchTrainer()
        {
            Layer layer = new Layer(new double[,] { { 1, 1 } }, new double[] { 0 }, TrainableNetworkComponent.TrainingMode.ONLINE);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            trainer.Train();
        }

        //[TestMethod]
        //public void BatchTrainCorrectOnePass()
        //{
        //    Layer layer = new Layer(new double[,] { { 1, 1 } });
        //    List<TrainingVector> trainingVectors = new List<TrainingVector>
        //    {
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 0 } ),
        //            new NetworkVector( new double[] { 1 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 0 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 1 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 1 } ),
        //            new NetworkVector( new double[] { 1 })
        //            )
        //    };

        //    BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
        //    trainer.Train();

        //    double[,] weightsCheck = new double[,] { { 1, 3 } };
        //    double[] biasesCheck = new double[] { 1 };

        //    for (int i = 0; i < layer.NumberOfOutputs; i++)
        //    {
        //        Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

        //        for (int j = 0; j < layer.NumberOfInputs; j++)
        //        {
        //            Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
        //        }

        //    }
        //}

        //[TestMethod]
        //public void TrainCorrectTwoPasses()
        //{
        //    Layer layer = new Layer(new double[,] { { 1, 1 } });
        //    List<TrainingVector> trainingVectors = new List<TrainingVector>
        //    {
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 0 } ),
        //            new NetworkVector( new double[] { 1 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 0 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 1 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 1 } ),
        //            new NetworkVector( new double[] { 1 })
        //            )
        //    };

        //    OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
        //    trainer.Train();
        //    trainer.Train();

        //    double[,] weightsCheck = new double[,] { { 3, 5 } };
        //    double[] biasesCheck = new double[] { 1 };

        //    for (int i = 0; i < layer.NumberOfOutputs; i++)
        //    {
        //        Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

        //        for (int j = 0; j < layer.NumberOfInputs; j++)
        //        {
        //            Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
        //        }

        //    }
        //}

        //[TestMethod]
        //public void TrainCorrectThreePasses()
        //{
        //    Layer layer = new Layer(new double[,] { { 1, 1 } });
        //    List<TrainingVector> trainingVectors = new List<TrainingVector>
        //    {
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 0 } ),
        //            new NetworkVector( new double[] { 1 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 0 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 0, 1 } ),
        //            new NetworkVector( new double[] { 0 })
        //            ),
        //        new TrainingVector(
        //            new NetworkVector( new double[] { 1, 1 } ),
        //            new NetworkVector( new double[] { 1 })
        //            )
        //    };

        //    OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
        //    trainer.Train();
        //    trainer.Train();
        //    trainer.Train();

        //    double[,] weightsCheck = new double[,] { { 3, 7 } };
        //    double[] biasesCheck = new double[] { -1 };

        //    for (int i = 0; i < layer.NumberOfOutputs; i++)
        //    {
        //        Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);

        //        for (int j = 0; j < layer.NumberOfInputs; j++)
        //        {
        //            Assert.AreEqual(weightsCheck[i, j], layer.State.Weights[i, j]);
        //        }

        //    }
        //}
    }
}

