using NeuralNet;

namespace FourthWord
{
    class FourthWordNetwork : NetComponentChain
    {
        private static int __inputs = 250;
        private static int __inputWords = 3;
        private static int __embeddingOutputs = 50;
        private static int __hiddenOutputs = 200;
        private static int __outputs = 250;

        #region private attributes
        private TrainableComponent _embeddingLayer;
        private TrainableComponent _hiddenLayer;
        private TrainableComponent _outputLayer;
        //private NetComponent _softMaxLayer;
        #endregion

        #region constructors
        public FourthWordNetwork()
        {
            WeightsMatrix embeddingWeights = MatrixProvider.GetRandom(__embeddingOutputs, __inputs);
            Layer embedding = Layer.CreateLinearLayer(embeddingWeights);
            _embeddingLayer = new TrainableComponentBank(embedding, __inputWords);

            WeightsMatrix hiddenWeights = MatrixProvider.GetRandom(__hiddenOutputs, __embeddingOutputs * __inputWords);
            _hiddenLayer = Layer.CreateLogisticLayer(hiddenWeights);

            WeightsMatrix outputweights = MatrixProvider.GetRandom(__outputs, __hiddenOutputs);
            _outputLayer = Layer.CreateLinearLayer(outputweights);

            //_softMaxLayer = new SoftMaxUnit(__outputs);
            
            this.AddTrainable(_embeddingLayer);
            this.AddTrainable(_hiddenLayer);
            this.AddTrainable(_outputLayer);
            //this.AddFixed(_softMaxLayer);
        }
        #endregion


    }
}
