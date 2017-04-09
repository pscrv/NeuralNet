using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2
{
    public abstract class Component
    {
        #region private attributes
        private int _numberOfInputs;
        private int _numberOfOutputs;
        #endregion


        #region constructors
        public Component(int numberOfInputs, int numberOfOutputs)
        {
            if (numberOfInputs < 1 || numberOfOutputs < 1)
                throw new ArgumentNullException("Any component must have at least one input and at least one output.");

            _numberOfInputs = numberOfInputs;
            _numberOfOutputs = numberOfOutputs;
        }
        #endregion


        #region public methods
        public DataVector Run(DataVector input)
        {
            return new DataVector( _run(input));
        }

        public VectorBatch Run(VectorBatch input)
        {
            return _run(input);
        }

        public DataVector BackPropagate(DataVector outputGradient)
        {
            return new DataVector(_backPropagate(outputGradient));
        }

        public VectorBatch BackPropagate(VectorBatch outputGradient)
        {
            return _backPropagate(outputGradient);
        }
        #endregion


        #region public properties
        public int NumberOfInputs { get { return _numberOfInputs; } }
        public int NumberOfOutputs { get { return _numberOfOutputs; } }
        #endregion


        #region abstract methods
        protected abstract VectorBatch _run(VectorBatch input);
        protected abstract VectorBatch _backPropagate(VectorBatch outputGradient);
        #endregion

    }
}
