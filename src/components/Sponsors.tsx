import { Cpu, Code, Shield, Cloud, Terminal, Layers } from "lucide-react";

interface SponsorProps {
  icon: JSX.Element;
  name: string;
  description: string;
}

const sponsors: SponsorProps[] = [
  {
    icon: <Cpu size={34} />,
    name: "AI & Machine Learning",
    description: "Leveraging AI algorithms to enhance real-time speech-to-sign and sign-to-speech translations for accurate communication.",
  },
  {
    icon: <Code size={34} />,
    name: "MERN Stack",
    description: "Utilizing MongoDB, Express.js, React, and Node.js for building a scalable and responsive web application.",
  },
  {
    icon: <Shield size={34} />,
    name: "Security & Privacy",
    description: "Implementing robust encryption protocols and secure authentication to ensure user privacy and data integrity.",
  },
  {
    icon: <Cloud size={34} />,
    name: "Cloud Infrastructure",
    description: "Harnessing cloud platforms for scalability, storage, and real-time data synchronization, ensuring performance and reliability.",
  },
  {
    icon: <Terminal size={34} />,
    name: "Version Control (Git)",
    description: "Using Git for version control to collaborate and manage the codebase with efficiency and ease.",
  },
  {
    icon: <Layers size={34} />,
    name: "Development Tools & Frameworks",
    description: "Using frameworks like Flask (Python) and TensorFlow for backend development and AI model training.",
  },
];

export const Sponsors = () => {
  return (
    <section id="sponsors" className="container pt-29 sm:py-32">
      <h1 className="text-center text-lg lg:text-4xl font-bold mb-10 text-primary">
        Technologies Used
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {sponsors.map(({ icon, name, description }: SponsorProps) => (
          <div
            key={name}
            className="flex flex-col items-center bg-transparent border-2 border-white/20 rounded-lg p-6 shadow-lg hover:shadow-xl transition-shadow duration-300"
          >
            <div className="mb-4">{icon}</div>
            <h3 className="text-xl font-semibold text-primary">{name}</h3>
            <p className="text-sm text-muted-foreground text-center mt-2">{description}</p>
          </div>
        ))}
      </div>
    </section>
  );
};
