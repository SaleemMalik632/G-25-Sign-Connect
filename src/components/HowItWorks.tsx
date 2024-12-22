import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { MedalIcon, MapIcon, PlaneIcon, GiftIcon } from "../components/Icons";

interface FeatureProps {
  icon: JSX.Element;
  title: string;
  description: string;
}

const features: FeatureProps[] = [
  {
    icon: <MedalIcon />,
    title: "Accessibility",
    description:
      "SignConnect ensures that individuals with hearing or speech disabilities can communicate easily through real-time speech-to-sign and sign-to-speech translation, making the world more inclusive.",
  },
  {
    icon: <MapIcon />,
    title: "Community Support",
    description:
      "We collaborate with the deaf and hard-of-hearing communities to ensure SignConnect is relevant and effective in bridging communication gaps across different languages and cultures.",
  },
  {
    icon: <PlaneIcon />,
    title: "Scalability",
    description:
      "SignConnect is built on scalable cloud infrastructure, allowing for easy expansion to serve users worldwide and supporting multiple languages for broader reach and impact.",
  },
  {
    icon: <GiftIcon />,
    title: "Real-time Interaction",
    description:
      "Our platform provides real-time translation, enabling seamless communication between sign language users and people without prior knowledge of sign language.",
  },
];

export const HowItWorks = () => {
  return (
    <section id="howItWorks" className="container text-center py-24 sm:py-32">
      <h2 className="text-3xl md:text-4xl font-bold ">
        How It{" "}
        <span className="bg-gradient-to-b from-primary/60 to-primary text-transparent bg-clip-text">
          Works{" "}
        </span>
        Step-by-Step Guide
      </h2>
      <p className="md:w-3/4 mx-auto mt-4 mb-8 text-xl text-muted-foreground">
        SignConnect revolutionizes communication for the hearing and speech-impaired. Here's how it works:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        {features.map(({ icon, title, description }: FeatureProps) => (
          <Card key={title} className="bg-muted/50 border-2 border-white/20 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300">
            <CardHeader>
              <CardTitle className="grid gap-4 place-items-center text-primary">
                {icon}
                <h3 className="text-lg font-semibold">{title}</h3>
              </CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">{description}</CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
};
