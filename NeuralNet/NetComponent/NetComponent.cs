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
        public abstract void Run(NetworkVector input);

        public abstract int NumberOfInputs { get; }
        public abstract int NumberOfOutputs { get; }
        public abstract NetworkVector Input { get; set; }
        public abstract NetworkVector Output { get; protected set; }
        #endregion    

    }    
}
