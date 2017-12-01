namespace TurtleML
{
    public interface ILayerBuilder
    {
        ILayer Build(ILayer inputLayer);
    }
}