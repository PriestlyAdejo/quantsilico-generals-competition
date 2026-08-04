import { useState, useRef, useCallback, useEffect } from "react";
import { MatchFrame } from "../types/match";

export function useReplayPlayer(frames: MatchFrame[], initialSpeed = 1) {
  const [currentTurn, setCurrentTurn] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(initialSpeed);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    setPlaying(false);
  }, []);

  const play = useCallback(() => {
    if (frames.length === 0) return;
    setPlaying(true);
    intervalRef.current = setInterval(() => {
      setCurrentTurn(t => {
        if (t >= frames.length - 1) { stop(); return t; }
        return t + 1;
      });
    }, Math.round(500 / speed));
  }, [frames.length, speed, stop]);

  useEffect(() => { if (playing) { stop(); play(); } }, [speed]);
  useEffect(() => () => stop(), []);

  const seekTo = useCallback((turn: number) => {
    stop();
    setCurrentTurn(Math.max(0, Math.min(turn, frames.length - 1)));
  }, [frames.length, stop]);

  const togglePlay = useCallback(() => {
    if (playing) stop();
    else play();
  }, [playing, play, stop]);

  const currentFrame = frames[currentTurn] ?? null;

  return { currentTurn, currentFrame, playing, speed, setSpeed, seekTo, togglePlay, stop, play };
}
