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
        public void CanMake()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunOnline_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };
        }

        [TestMethod]
        public void CanRunOnline_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector(new double[] { 0, 0 }),
                    new NetworkVector(new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(layer, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);
        }

        [TestMethod]
        public void CanRunOnline_LogisticLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLogisticLayer(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                 new VectorPair(
                    new NetworkVector(new double[] { 0, 0 }),
                    new NetworkVector(new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(layer, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);
        }



        #region online training tests
        [TestMethod]
        public void TrainOnline_LinearLayer_CorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(layer, new SquaredError(), new GradientDescent());
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 1, 3 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 1 });

            Assert.AreEqual(biasesCheck, layer.Biases);
            Assert.AreEqual(weightsCheck, layer.Weights);
        }

        [TestMethod]
        public void TrainOnline_WC_CorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0 } ), 
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }
            
            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 1, 3 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 1 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void TrainOnline_WC_CorrectTwoPasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 5 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { 1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }
        
        [TestMethod]
        public void TrainOnline_WC_CorrectThreePasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 7 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { -1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);            
        }

        [TestMethod]
        public void TrainOnline_SmallChain_CorrectOnePass()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;

            double[,] inputWeights = new double[inputneurons, inputs];
            double[,] outputWeights = new double[outputneurons, inputneurons];

            for (int i = 0; i < inputneurons; i++)
                for (int j = 0; j < inputs; j++)
                    inputWeights[i, j] = 1;

            for (int i = 0; i < outputneurons; i++)
                for (int j = 0; j < inputneurons; j++)
                    outputWeights[i, j] = 1;
            
            Layer InputLayer = Layer.CreateLinearLayer(new NetworkMatrix(inputWeights), new NetworkVector(inputneurons));
            Layer OutputLayer = Layer.CreateLinearLayer(new NetworkMatrix(outputWeights), new NetworkVector(outputneurons));

            NetComponentChain network = new NetComponentChain();
            network.AddTrainable(InputLayer);
            network.AddTrainable(OutputLayer);

            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(network, new SquaredError(), new GradientDescent());
            foreach (TrainingCollection tc in trainingVectors.AsSingletons())
            {
                trainer.Train(tc);
            }

            NetworkMatrix inputWeightsCheck = new NetworkMatrix(new double[,] { { -35499715, -35499260, 1 }, { -35499715, -35499260, 1 } });
            NetworkVector inputBiasesCheck = new NetworkVector(new double[] { -35499265, -35499265 });
            NetworkMatrix outputWeightsCheck = new NetworkMatrix(new double[,] { { -224831362, -224831362 } });
            NetworkVector outputBiasesCheck = new NetworkVector(new double[] { -251825 });

            Assert.AreEqual(inputWeightsCheck, InputLayer.Weights);
            Assert.AreEqual(inputBiasesCheck, InputLayer.Biases);
            Assert.AreEqual(outputWeightsCheck, OutputLayer.Weights);
            Assert.AreEqual(outputBiasesCheck, OutputLayer.Biases);
        }
        #endregion



        #region batch training tests
        [TestMethod]
        public void BatchTrainCorrectOnePass_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
        {
            new VectorPair(
                new NetworkVector( new double[] { 0, 0 } ),
                new NetworkVector( new double[] { 1 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 0 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 0, 1 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 1 } ),
                new NetworkVector( new double[] { 1 })
                )
        };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { -1, -1 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { -2 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectOnePass_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
        {
            new VectorPair(
                new NetworkVector( new double[] { 0, 0 } ),
                new NetworkVector( new double[] { 1 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 0 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 0, 1 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 1 } ),
                new NetworkVector( new double[] { 1 })
                )
        };

            Trainer trainer = new Trainer(layer, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { -1, -1 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { -2 });

            Assert.AreEqual(biasesCheck, layer.Biases);
            Assert.AreEqual(weightsCheck, layer.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectTwoPasses_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
        {
            new VectorPair(
                new NetworkVector( new double[] { 0, 0 } ),
                new NetworkVector( new double[] { 1 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 0 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 0, 1 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 1 } ),
                new NetworkVector( new double[] { 1 })
                )
        };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);
            trainer.Train(trainingVectors);

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 7, 7 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 12 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectThreePasses_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            TrainingCollection trainingVectors = new TrainingCollection
        {
            new VectorPair(
                new NetworkVector( new double[] { 0, 0 } ),
                new NetworkVector( new double[] { 1 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 0 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 0, 1 } ),
                new NetworkVector( new double[] { 0 })
                ),
            new VectorPair(
                new NetworkVector( new double[] { 1, 1 } ),
                new NetworkVector( new double[] { 1 })
                )
        };

            Trainer trainer = new Trainer(wc, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);
            trainer.Train(trainingVectors);
            trainer.Train(trainingVectors);

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { -37, -37 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { -62 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void TrainBatch_SmallChain_CorrectOnePass()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;

            double[,] inputWeights = new double[inputneurons, inputs];
            double[,] outputWeights = new double[outputneurons, inputneurons];

            for (int i = 0; i < inputneurons; i++)
                for (int j = 0; j < inputs; j++)
                    inputWeights[i, j] = 1;

            for (int i = 0; i < outputneurons; i++)
                for (int j = 0; j < inputneurons; j++)
                    outputWeights[i, j] = 1;

            Layer InputLayer = Layer.CreateLinearLayer(new NetworkMatrix(inputWeights), new NetworkVector(inputneurons));
            Layer OutputLayer = Layer.CreateLinearLayer(new NetworkMatrix(outputWeights), new NetworkVector(outputneurons));

            NetComponentChain network = new NetComponentChain();
            network.AddTrainable(InputLayer);
            network.AddTrainable(OutputLayer);


            TrainingCollection trainingVectors = new TrainingCollection
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            Trainer trainer = new Trainer(network, new SquaredError(), new GradientDescent());
            trainer.Train(trainingVectors);

            NetworkMatrix inputWeightsCheck = new NetworkMatrix(new double[,] { { -4, -4, 1 }, { -4, -4, 1 } });
            NetworkVector inputBiasesCheck = new NetworkVector(new double[] { -6, -6 });
            NetworkMatrix outputWeightsCheck = new NetworkMatrix(new double[,] { { -9, -9 } });
            NetworkVector outputBiasesCheck = new NetworkVector(new double[] { -6 });

            Assert.AreEqual(inputWeightsCheck, InputLayer.Weights);
            Assert.AreEqual(inputBiasesCheck, InputLayer.Biases);
            Assert.AreEqual(outputWeightsCheck, OutputLayer.Weights);
            Assert.AreEqual(outputBiasesCheck, OutputLayer.Biases);
        }
        #endregion
    }
}

