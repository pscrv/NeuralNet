using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class WeightedCombiner : TrainableNetworkComponent
    {
        #region private attributes
        protected NetworkVector _inputs;
        protected NetworkMatrix _weights;
        protected NetworkVector _biases;
        protected NetworkVector _outputs;
        protected NetworkVector _inputGradient;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _weights.NumberOfOutputs; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        public LayerState State { get { return new LayerState(_weights, _biases); } }
        #endregion

        #region Constructors
        public WeightedCombiner(NetworkMatrix weights, NetworkVector biases, TrainingMode mode)
            : base (mode)
        {
            if (weights == null)
                throw new ArgumentException("Attempt to create a WeightedCombiner with null weights.");

            if (biases == null)
                throw new ArgumentException("Attempt to create a WeightedCombiner with null biases.");

            if (biases.Dimension != weights.NumberOfOutputs)
                throw new ArgumentException("Dimension of biases must the the same of the outputs.");

            _weights = weights.Copy();
            _biases = biases.Copy();

            _inputs = new NetworkVector(NumberOfInputs);
            _outputs = new NetworkVector(NumberOfOutputs);
            _inputGradient = new NetworkVector(NumberOfInputs);
        }

        public WeightedCombiner(NetworkMatrix weights, NetworkVector biases)
            : this(weights, biases, TrainingMode.ONLINE) { }

        public WeightedCombiner(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }

        
        public WeightedCombiner(double[,] weights, double[] biases)
            : this (new NetworkMatrix(weights), new NetworkVector(biases)) { }

        public WeightedCombiner(double[,] weights)
            : this(new NetworkMatrix(weights)) { }

        public WeightedCombiner(WeightedCombiner combiner)
        {
            if (combiner == null)
                throw new ArgumentException("Attempt to make a WeightedCombiner from null");

            this._biases = combiner._biases.Copy();
            this._weights = combiner._weights.Copy();
            this.Mode = combiner.Mode;
        }
        #endregion

        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");
            _inputs = inputvalues;
            _outputs = _biases.SumWith(_weights.LeftMultiply(_inputs));

        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            _inputGradient = _weights.DotWithWeightsPerInput(outputgradient);
            _biases.Subtract(outputgradient);
            _weights.Subtract(outputgradient.LeftMultiply(_inputs));
        }
        #endregion
    }



    public class OnlineWeightedCombiner : WeightedCombiner
    {
        #region Constructors
        public OnlineWeightedCombiner(NetworkMatrix weights, NetworkVector biases)
            : base (weights, biases)
        { }

        public OnlineWeightedCombiner(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }


        public OnlineWeightedCombiner(double[,] weights, double[] biases)
            : this (new NetworkMatrix(weights), new NetworkVector(biases)) { }

        public OnlineWeightedCombiner(double[,] weights)
            : this(new NetworkMatrix(weights)) { }

        public OnlineWeightedCombiner(OnlineWeightedCombiner combiner)
            : base(combiner)
        { }
        #endregion
    }



    public class BatchWeightedCombiner : WeightedCombiner
    {
        #region private attributes
        protected NetworkVector _biasesDelta;
        protected NetworkMatrix _weightsDelta;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _weights.NumberOfOutputs; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        public LayerState State { get { return new LayerState(_weights, _biases); } }
        #endregion

        #region Constructors
        public BatchWeightedCombiner(NetworkMatrix weights, NetworkVector biases)
            : base (weights, biases)
        {
            _biasesDelta = new NetworkVector(NumberOfOutputs);
            _weightsDelta = new NetworkMatrix(NumberOfOutputs, NumberOfInputs);        
        }
        
        public BatchWeightedCombiner(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }

        public BatchWeightedCombiner(double[,] weights, double[] biases)
            : this (new NetworkMatrix(weights), new NetworkVector(biases)) { }

        public BatchWeightedCombiner(double[,] weights)
            : this(new NetworkMatrix(weights)) { }

        public BatchWeightedCombiner(BatchWeightedCombiner combiner)
            :this (combiner?._weights.Copy(), combiner?._biases.Copy()) { }
        #endregion


        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");
            _inputs = inputvalues;
            _outputs = _biases.SumWith(_weights.LeftMultiply(_inputs));

        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            _inputGradient = _weights.DotWithWeightsPerInput(outputgradient);
            
            _biasesDelta.Add(outputgradient);
            _weightsDelta.Add(outputgradient.LeftMultiply(_inputs));             
        }

        public void StartBatch()
        {
            _biasesDelta.Zero();
            _weightsDelta.Zero();
        }

        public void EndBatchAndUpdate()
        {
            _biases.Subtract(_biasesDelta);
            _weights.Subtract(_weightsDelta);
            StartBatch();
        }

        public void DiscardBatch()
        {
            StartBatch();
        }
        #endregion
    }

}