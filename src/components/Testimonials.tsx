import { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FaMicrophone, FaMicrophoneSlash, FaVideo, FaVideoSlash } from "react-icons/fa";  // Importing icons

// TestimonialProps and sample testimonials as before
interface TestimonialProps {
  image: string;
  name: string;
  userName: string;
  comment: string;
}

const testimonials: TestimonialProps[] = [
  {
    image: "https://github.com/shadcn.png",
    name: "John Doe React",
    userName: "@john_Doe",
    comment: "This landing page is awesome!",
  },
];

export const Testimonials = () => {
  const [meetingCode, setMeetingCode] = useState<string>("");
  const [meetingStatus, setMeetingStatus] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [micMuted, setMicMuted] = useState<boolean>(false);
  const [cameraOff, setCameraOff] = useState<boolean>(false);

  const handleJoinMeeting = () => {
    if (meetingCode) {
      setIsModalOpen(true); // Show the modal
      setMeetingStatus("Please wait while we are connecting...");
      
      // Simulate the connection process (e.g., 3 seconds)
      setTimeout(() => {
        setMeetingStatus("Successfully connected!");
      }, 3000);
    } else {
      setMeetingStatus("Please enter a valid meeting code.");
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setMeetingStatus("");
  };

  return (
    <section id="meeting" className="container py-24 sm:py-32 rounded-xl shadow-lg">
      <h2 className="text-3xl md:text-4xl font-bold text-center mb-8">
        Discover Why
        <span className="bg-gradient-to-b from-primary/60 to-primary text-transparent bg-clip-text">
          {" "}
          People Love{" "}
        </span>
        This Meeting Experience
      </h2>

      <p className="text-xl text-muted-foreground text-center pt-4 pb-8">
        See what our community members are saying about their experiences
        with our platform.
      </p>

      {/* Centered Box with Input and Button */}
      <div className="flex justify-center items-center">
        <Card className="max-w-md w-full p-6 border border-gray-200 rounded-lg shadow-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-lg font-semibold">Join a Meeting</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Meeting Code Input */}
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
            {/* Join Button */}
            <button
              onClick={handleJoinMeeting}
              className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
            >
              Join Meeting
            </button>
            {/* Meeting Status */}
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
              {/* Video Screen Placeholder */}
              <div className="bg-gray-800 w-full h-72 rounded-lg flex justify-center items-center text-white">
                <p className="text-lg">Video Screen</p>
              </div>
            </div>
            {/* Control buttons */}
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
            {/* Connection Status */}
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
