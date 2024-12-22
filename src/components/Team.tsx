import { buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Facebook, Instagram, Linkedin } from "lucide-react";
import Mustafa from "../assets/Team Images/Mustafa.png";
import Saleem from "../assets/Team Images/Saleem.png";
import Abdulmannan from "../assets/Team Images/Abdulmannan.png";
import Abdurehman from "../assets/Team Images/Abdurehman.png";

interface TeamProps {
  imageUrl: string;
  name: string;
  position: string;
  socialNetworks: SociaNetworkslProps[];
}

interface SociaNetworkslProps {
  name: string;
  url: string;
}

const teamList: TeamProps[] = [
  {
    imageUrl: "https://media.licdn.com/dms/image/v2/D4D03AQF9x1BL4EOZYQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1724423918811?e=1740614400&v=beta&t=hW872HlBXZmzNXFja2yuXTIYslTRxV_XC58E5Kn06KM",
    name: "Waqas Ali",
    position: "Supervisor",
    socialNetworks: [
      { name: "Linkedin", url: "https://www.linkedin.com/in/leopoldo-miranda/" },
      { name: "Facebook", url: "https://www.facebook.com/" },
      { name: "Instagram", url: "https://www.instagram.com/" },
    ],
  },
  {
    imageUrl: Saleem,
    name: "Saleem Malik",
    position: "Team Lead",
    socialNetworks: [
      { name: "Linkedin", url: "https://www.linkedin.com/in/leopoldo-miranda/" },
      { name: "Facebook", url: "https://www.facebook.com/" },
      { name: "Instagram", url: "https://www.instagram.com/" },
    ],
  },
  {
    imageUrl: Mustafa,
    name: "Mustafa Riaz",
    position: "Full Stack Developer",
    socialNetworks: [
      { name: "Linkedin", url: "https://www.linkedin.com/in/mustafa-riaz-dev/" },
      { name: "Facebook", url: "https://www.facebook.com/" },
      { name: "Instagram", url: "https://www.instagram.com/" },
    ],
  },
  {
    imageUrl: Abdulmannan,
    name: "Abdul Mannan",
    position: "AI Developer",
    socialNetworks: [
      { name: "Linkedin", url: "https://www.linkedin.com/in/leopoldo-miranda/" },
      { name: "Instagram", url: "https://www.instagram.com/" },
    ],
  },
  {
    imageUrl: Abdurehman,
    name: "Abdurehman",
    position: "Software Tester",
    socialNetworks: [
      { name: "Linkedin", url: "https://www.linkedin.com/in/leopoldo-miranda/" },
      { name: "Facebook", url: "https://www.facebook.com/" },
    ],
  },
];

export const Team = () => {
  const socialIcon = (iconName: string) => {
    switch (iconName) {
      case "Linkedin":
        return <Linkedin size="20" />;
      case "Facebook":
        return <Facebook size="20" />;
      case "Instagram":
        return <Instagram size="20" />;
    }
  };

  return (
    <section id="team" className="container py-24 sm:py-32">
      <h2 className="text-3xl md:text-4xl font-bold">
        <span className="bg-gradient-to-b from-primary/60 to-primary text-transparent bg-clip-text">
          Our Dedicated{" "}
        </span>
        Crew
      </h2>

      <p className="mt-4 mb-10 text-xl text-muted-foreground">
        Meet the passionate team behind SignConnect! Our team is committed to providing innovative solutions for seamless communication for people with hearing and speech disabilities.
      </p>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 gap-y-10">
        {teamList.map(
          ({ imageUrl, name, position, socialNetworks }: TeamProps) => (
            <Card
              key={name}
              className="bg-muted/50 relative mt-8 flex flex-col justify-center items-center"
            >
              <CardHeader className="mt-8 flex justify-center items-center pb-2">
                <img
                  src={imageUrl}
                  alt={`${name} ${position}`}
                  className="absolute -top-12 rounded-full w-24 h-24 aspect-square object-cover"
                />
                <CardTitle className="text-center">{name}</CardTitle>
                <CardDescription className="text-primary">
                  {position}
                </CardDescription>
              </CardHeader>

              <CardContent className="text-center pb-2">
                <p>{name} is a dedicated member of our team, bringing expertise in {position.toLowerCase()} to make SignConnect a revolutionary platform for accessibility.</p>
              </CardContent>

              <CardFooter>
                {socialNetworks.map(({ name, url }: SociaNetworkslProps) => (
                  <div key={name}>
                    <a
                      rel="noreferrer noopener"
                      href={url}
                      target="_blank"
                      className={buttonVariants({
                        variant: "ghost",
                        size: "sm",
                      })}
                    >
                      <span className="sr-only">{name} icon</span>
                      {socialIcon(name)}
                    </a>
                  </div>
                ))}
              </CardFooter>
            </Card>
          )
        )}
      </div>
    </section>
  );
};
