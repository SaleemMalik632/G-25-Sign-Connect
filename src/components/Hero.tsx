import { useNavigate } from "react-router-dom";
import { Button } from "./ui/button";
import { buttonVariants } from "./ui/button";
import { HeroCards } from "./HeroCards";
import { GitHubLogoIcon } from "@radix-ui/react-icons";

export const Hero = () => {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName"); // Get userName from localStorage

  // Handle login/logout behavior
  const handleGetStarted = () => {
    if (userName) {
      // If the user is logged in, handle logout
      localStorage.removeItem("userName");
      navigate("/"); // Optionally navigate to the home page after logout
    } else {
      // If the user is not logged in, navigate to the login page
      navigate("/getstarted");
    }
  };

  return (
    <section className="container grid lg:grid-cols-2 place-items-center py-20 md:py-32 gap-10">
      <div className="text-center lg:text-start space-y-6">
        <main className="text-5xl md:text-6xl font-bold">
          <h1 className="inline">
            <span className="inline bg-gradient-to-r from-[#F596D3]  to-[#D247BF] text-transparent bg-clip-text">
              SignConnect
            </span>{" "}
            {/* updated name */}
          </h1>{" "}
          {/* heading for the project */}
        </main>

        <p className="text-xl text-muted-foreground md:w-10/12 mx-auto lg:mx-0">
          SignConnect is an AI-powered application designed to bridge communication gaps for people with hearing and speech disabilities. It helps translate sign language to speech and vice versa, enabling seamless interaction with the world.
        </p>

        {/* Conditionally render the welcome message and buttons */}
        {userName ? (
          <div className="space-y-4 md:space-y-0 md:space-x-4">
            {/* Welcome message only shown if user is logged in */}
            <Button className="w-full md:w-1/3" onClick={handleGetStarted}>
              Logout
            </Button>
            <a
              rel="noreferrer noopener"
              href="https://github.com/alanjeremiah/WLASL-Recognition-and-Translation"
              target="_blank"
              className={`w-full md:w-1/3 ${buttonVariants({
                variant: "outline",
              })}`}
            >
              Github Repository
              <GitHubLogoIcon className="ml-2 w-5 h-5" />
            </a>
          </div>
        ) : (
          <div className="space-y-4 md:space-y-0 md:space-x-4">
            {/* Show Login button if no user is logged in */}
            <Button className="w-full md:w-1/3" onClick={handleGetStarted}>
              Login
            </Button>
            <a
              rel="noreferrer noopener"
              href="https://github.com/alanjeremiah/WLASL-Recognition-and-Translation"
              target="_blank"
              className={`w-full md:w-1/3 ${buttonVariants({
                variant: "outline",
              })}`}
            >
              Github Repository
              <GitHubLogoIcon className="ml-2 w-5 h-5" />
            </a>
          </div>
        )}
      </div>

      {/* Hero cards sections */}
      <div className="z-10">
        <HeroCards />
      </div>

      {/* Shadow effect */}
      <div className="shadow"></div>
    </section>
  );
};
