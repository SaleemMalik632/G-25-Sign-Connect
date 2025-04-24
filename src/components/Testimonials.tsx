import { useState, useRef, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  FaMicrophone,
  FaMicrophoneSlash,
  FaVideo,
  FaVideoSlash,
} from "react-icons/fa";

export const Testimonials = () => {
  const [meetingCode, setMeetingCode] = useState<string>("");
  const [meetingStatus, setMeetingStatus] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [micMuted, setMicMuted] = useState<boolean>(false);
  const [cameraOff, setCameraOff] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (isModalOpen && !cameraOff) {
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        })
        .catch((err) => {
          console.error("Error accessing media devices:", err);
          setMeetingStatus("Failed to access camera or microphone.");
        });
    }
  }, [isModalOpen, cameraOff]);

  const handleJoinMeeting = () => {
    if (meetingCode) {
      setIsModalOpen(true);
      setMeetingStatus("Please wait while we are connecting...");

      setTimeout(() => {
        setMeetingStatus("Successfully connected!");
      }, 3000);
    } else {
      setMeetingStatus("Please enter a valid meeting code.");
    }
  };

  const handleCloseModal = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsModalOpen(false);
    setMeetingStatus("");
  };

  return (
    <section id="meeting" className="container py-24 sm:py-32 rounded-xl shadow-lg">
      <h2 className="text-3xl md:text-4xl font-bold text-center mb-8">
        Discover Why
        <span className="bg-gradient-to-b from-primary/60 to-primary text-transparent bg-clip-text">
          {" "}People Love{" "}
        </span>
        This Meeting Experience
      </h2>

      <p className="text-xl text-muted-foreground text-center pt-4 pb-8">
        See what our community members are saying about their experiences with our platform.
      </p>

      {/* Centered Box with Input and Button */}
      <div className="flex justify-center items-center">
        <Card className="max-w-md w-full p-6 border border-gray-200 rounded-lg shadow-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-lg font-semibold">Join a Meeting</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <label htmlFor="meetingCode" className="block text-sm font-medium text-white-700">
                Meeting Code
              </label>
              <input
                id="meetingCode"
                type="text"
                value={meetingCode}
                onChange={(e) => setMeetingCode(e.target.value)}
                className="mt-1 p-2 w-full border border-gray-300 rounded-md text-black-500 bg-transparent"
                placeholder="Enter your meeting code"
              />
            </div>
            <button
              onClick={handleJoinMeeting}
              className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
            >
              Join Meeting
            </button>
            {meetingStatus && (
              <p className="mt-4 text-center text-gray-700">{meetingStatus}</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Modal - Video Meeting Interface */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-gray-900 p-8 rounded-lg shadow-xl w-3/4 lg:w-1/2">
            <h3 className="text-xl font-semibold text-center text-white">Meeting in Progress</h3>
            <div className="flex justify-center mt-6">
              <div className="bg-gray-800 w-full h-72 rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover rounded-lg"
                  autoPlay
                  muted
                />
              </div>
            </div>
            <div className="flex justify-center gap-8 mt-8">
              <div className="flex flex-col items-center">
                <button
                  onClick={() => setMicMuted(!micMuted)}
                  className="text-white hover:text-gray-300"
                >
                  {micMuted ? <FaMicrophoneSlash size={24} /> : <FaMicrophone size={24} />}
                  <p className="text-sm">Mute Mic</p>
                </button>
              </div>
              <div className="flex flex-col items-center">
                <button
                  onClick={() => setCameraOff(!cameraOff)}
                  className="text-white hover:text-gray-300"
                >
                  {cameraOff ? <FaVideoSlash size={24} /> : <FaVideo size={24} />}
                  <p className="text-sm">Mute Camera</p>
                </button>
              </div>
            </div>
            <p className="mt-4 text-center text-gray-300">{meetingStatus}</p>
            <div className="flex justify-center mt-6">
              <button
                onClick={handleCloseModal}
                className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600"
              >
                End Meeting
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
