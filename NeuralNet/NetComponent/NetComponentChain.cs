using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{

    public class NetComponentChain : NetComponent, ITrainable
    {
        #region protected members
        protected _networkComponentNode _head;
        protected _networkComponentNode _tail;
        #endregion

        #region constructors
        public NetComponentChain()
        {
            _head = null;
            _tail = null;
        }

        public NetComponentChain(NetComponent component)
            : this()
        {
            AddFixed(component);
        }

        public NetComponentChain(TrainableComponent component)
            : this()
        {
            AddTrainable(component);
        }
        #endregion



        #region NetComponent overrides
        public override int NumberOfInputs { get { return _head.Component.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _tail.Component.NumberOfOutputs; } }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to back propogate in a network with no layers.");

            if (outputgradient.Dimension != _tail.Component.NumberOfOutputs)
                throw new ArgumentException(string.Format("The network has {0} outputs, but outputgradient has dimension {1}", _tail.Component.NumberOfOutputs, outputgradient.Dimension));

            NetworkVector gradient = outputgradient.Copy();
            foreach (NetComponent component in BackwardsEnumeration)
            {
                gradient = component.InputGradient(gradient);
            }

            return gradient;
        }

        public override VectorBatch InputGradient(VectorBatch outputgradients)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to back propogate in a network with no layers.");

            if (outputgradients.Dimension != _tail.Component.NumberOfOutputs)
                throw new ArgumentException(string.Format("The network has {0} outputs, but outputgradient has dimension {1}", _tail.Component.NumberOfOutputs, outputgradients.Dimension));

            VectorBatch gradient = outputgradients;
            foreach (NetComponent component in BackwardsEnumeration)
            {
                gradient = component.InputGradient(gradient);
            }

            return gradient;
        }

        public override NetworkVector Run(NetworkVector input)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to run a network with no layers.");

            if (input.Dimension != NumberOfInputs)
                throw new ArgumentException(string.Format("The network accepts {0} inputs, but input has dimension {1}", NumberOfInputs, input.Dimension));

            NetworkVector result = input;
            foreach (NetComponent component in ForwardEnumeration)
            {
                result = component.Run(result);
            }

            return result;
        }

        public override VectorBatch Run(VectorBatch inputbatch)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to run a network with no layers.");

            if (inputbatch.Dimension != NumberOfInputs)
                throw new ArgumentException(string.Format("The network accepts {0} inputs, but input has dimension {1}", NumberOfInputs, inputbatch.Dimension));

            VectorBatch result = inputbatch;
            foreach (NetComponent component in ForwardEnumeration)
            {
                result = component.Run(result);
            }

            return result;
        }
        #endregion


        #region public methods
        public int NumberOfComponents
        {
            get
            {
                if (_head == null)
                    return 0;

                if (_head == _tail)
                    return 1;

                int count = 0;
                _networkComponentNode node = _head;
                while (node != null)
                {
                    count++;
                    node = node.Next;
                }
                return count;

            }
        }

        public void AddFixed(NetComponent componentToAdd)
        {
            _addComponent(componentToAdd, istrainable: false);
        }

        public void AddTrainable(TrainableComponent componentToAdd)
        {
            _addComponent(componentToAdd, istrainable: true);
        }

        #endregion


        #region protected and private methods
        protected void _addComponent(NetComponent componentToAdd, bool istrainable)
        {
            if (_componentHasWrongSize(componentToAdd))
                throw new ArgumentException("Attempt to add with inputs size differnt from output size of the last existing Layer");

            _networkComponentNode nodeToAdd = new _networkComponentNode(componentToAdd, istrainable);

            if (_head == null)
            {
                _head = nodeToAdd;
                _tail = _head;
            }
            else
            {
                _tail.Next = nodeToAdd;
                _tail.Next.Previous = _tail;
                _tail = _tail.Next;
            }
        }

        protected bool _componentHasWrongSize(NetComponent componentToAdd)
        {
            if (NumberOfComponents == 0)
                return false;

            return _tail.Component.NumberOfOutputs != componentToAdd.NumberOfInputs;
        }
        #endregion


        #region ITrainable
        public void BackPropagate(NetworkVector outputgradient)
        {
            NetworkVector currentGradient = outputgradient.Copy();
            NetComponent currentComponent;

            _networkComponentNode node = _tail;
            {
                while (node != null)
                {
                    currentComponent = node.Component;
                    if (node.IsTrainable)
                    {
                        (currentComponent as TrainableComponent).BackPropagate(currentGradient);
                    }

                    currentGradient = currentComponent.InputGradient(currentGradient);
                    node = node.Previous;
                }
            }
        }

        public void BackPropagate(VectorBatch outputgradients)
        {
            VectorBatch currentGradient = outputgradients;
            NetComponent currentComponent;

            _networkComponentNode node = _tail;
            {
                while (node != null)
                {
                    currentComponent = node.Component;
                    if (node.IsTrainable)
                    {
                        (currentComponent as TrainableComponent).BackPropagate(currentGradient);
                    }

                    currentGradient = currentComponent.InputGradient(currentGradient);
                    node = node.Previous;
                }
            }
        }

        public void Update(AdaptationStrategy strategy)
        {
            foreach (TrainableComponent component in BackwardsTrainableComponentEnumeration)
            {
                component.Update(strategy);
            }
        }
        #endregion

        #region IEnumerable
        public IEnumerable<NetComponent> ForwardEnumeration
        {
            get
            {
                _networkComponentNode node = _head;

                while (node != null)
                {
                    yield return node.Component;
                    node = node.Next;
                }
            }
        }

        public IEnumerable<NetComponent> ForwardTrainableComponentsEnumeration
        {
            get
            {
                _networkComponentNode node = _head;

                while (node != null)
                {
                    if (node.IsTrainable)
                        yield return node.Component;
                    node = node.Next;
                }
            }
        }

        public IEnumerable<NetComponent> BackwardsEnumeration
        {
            get
            {
                _networkComponentNode node = _tail;

                while (node != null)
                {
                    yield return node.Component;
                    node = node.Previous;
                }
            }
        }

        public IEnumerable<TrainableComponent> BackwardsTrainableComponentEnumeration
        {
            get
            {
                _networkComponentNode node = _tail;

                while (node != null)
                {
                    if (node.IsTrainable)
                        yield return node.Component as TrainableComponent;
                    node = node.Previous;
                }
            }
        }      
        #endregion




        #region protected internal of LayerListNode
        protected class _networkComponentNode
        {
            public NetComponent Component { get; protected set; }
            public bool IsTrainable { get; protected set; }
            public _networkComponentNode Next { get; set; }
            public _networkComponentNode Previous { get; set; }

            public _networkComponentNode(NetComponent component, bool istrainable)
            {
                Component = component;
                IsTrainable = istrainable;
            }
        }
        #endregion
    }
}
