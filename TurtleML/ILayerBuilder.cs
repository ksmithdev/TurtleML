namespace TurtleML
{
    public interface ILayerBuilder
    {
        ILayer Build(IOutput input);
    }
}