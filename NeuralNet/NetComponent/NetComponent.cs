using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class NetComponent
    {
        #region abstract methods / properties
        public abstract NetworkVector InputGradient(NetworkVector outputgradient);
        public abstract VectorBatch InputGradient(VectorBatch outputgradient);
        public abstract NetworkVector Run(NetworkVector input);
        public abstract VectorBatch Run(VectorBatch inputbatch);

        //public abstract NetworkVector VectorInput { get; protected set; }

        public abstract int NumberOfInputs { get; }
        public abstract int NumberOfOutputs { get; }
        #endregion    

    }    
}
