import { useCanvasAnimation } from '@/hooks/useCanvasAnimation';

export const CanvasBackground = () => {
  const canvasRef = useCanvasAnimation();

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ opacity: 0.4 }}
      />
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/3 left-1/3 w-64 h-64 bg-lii-gold rounded-full blur-[120px] animate-[luxuryPulse_10s_ease-in-out_infinite]"></div>
      </div>
    </>
  );
};
