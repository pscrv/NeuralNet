using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LayerChain : NetworkComponentChain
    {
        #region public properties
        public List<LayerState> State
        {
            get
            {
                List<LayerState> state = new List<LayerState>();
                foreach (Layer layer in ForwardEnumeration)
                    state.Add(layer.State);
                return state;
            }
        }
        #endregion


        #region constructors
        public LayerChain()
            : base() { }

        public LayerChain(Layer layer)
            : base(layer) { }
        #endregion


        #region public methods
        public void Add(Layer layerToAdd)
        {
            base.Add(layerToAdd);
        }

        public override void Add(NetworkComponent componentToAdd)
        {
            Layer layerToAdd = componentToAdd as Layer;
            if (layerToAdd == null)
                throw new ArgumentException("Attempt to add non-layer to a LayerChain.");
            this.Add(layerToAdd);
        }

        #endregion
    }
}